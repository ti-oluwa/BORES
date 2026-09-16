"""Run a simulation workflow on a multi-dimensional reservoir model."""

import copy
import logging
import typing
from contextlib import ExitStack
from datetime import datetime, timezone
from os import PathLike

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.tables.pvt import PVTDataSet, PVTTables
from bores.boundary_conditions import BoundaryConditions, build_boundary_metadata
from bores.config import Config
from bores.constants import c
from bores.datastructures import (
    BottomHolePressure,
    BottomHolePressures,
    ContextFlag,
    FormationVolumeFactors,
    Rates,
    SparseTensor,
)
from bores.errors import SimulationError, StopSimulation, TimingError, ValidationError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids, RelPermGrids
from bores.grids.rock_fluid import (
    build_rock_fluid_properties_grids,
    build_three_phase_relative_mobilities_grids,
)
from bores.initialization import (
    apply_minimum_injector_saturations,
    seed_injection_saturations,
)
from bores.material_balance import (
    MaterialBalanceErrors,
    compute_material_balance_errors,
)
from bores.precision import get_dtype
from bores.reservoir import (
    BlackOil,
    FluidProperties,
    Hysteresis,
    RockProperties,
)
from bores.rock_fluid.relperm import RelPermEndpoints
from bores.serde.stores import StoreSerializable
from bores.solvers import explicit, implicit
from bores.solvers.base import normalize_saturations
from bores.solvers.rates import WellRates, compute_well_rates
from bores.states import ModelState, RatesInfo
from bores.transmissibility import FaceTransmissibilities
from bores.types import MiscibilityModel, NDimension, NDimensionalGrid, ThreeDimensions
from bores.updates import update_fluid_properties, update_residual_saturation_grids
from bores.validation import ValidationReport, validate
from bores.wells.base import Wells
from bores.wells.indices import WellsIndices, build_wells_indices, update_wells_indices

__all__ = ["Run", "StepResult", "run", "running", "stop"]

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
    time_step: int
    """The time step number (iteration index) for this result."""
    time_step_size: float
    """Size of the current time step (seconds)."""
    time: float
    """Total elapsed simulation time (seconds)."""
    hysteresis_state: Hysteresis[NDimension] | None = None
    """Updated hysteresis state after the time step."""
    rates: WellRates[NDimension] | None = None
    """Well rate info for the step"""
    success: bool = True
    """Whether the time step evolution was successful."""
    message: str | None = None
    """Optional message providing additional information about the time step result."""
    material_balance_errors: MaterialBalanceErrors | None = None
    """Material balance errors for this time step. None for rejected steps or first step."""
    maximum_pressure_change: float | None = None
    """Maximum pressure change from the pressure solution."""
    maximum_oil_saturation_change: float | None = None
    """Maximum oil saturation change from the transport solution."""
    maximum_water_saturation_change: float | None = None
    """Maximum water saturation change from the transport solution."""
    maximum_gas_saturation_change: float | None = None
    """Maximum gas saturation change from the transport solution."""
    timer_context: dict[str, typing.Any] = attrs.field(factory=dict)
    """Keyword argument that should be passed to the simulation timer on accepting or rejecting a step."""


@attrs.frozen(slots=True)
class SaturationChangeCheckResult:
    violated: bool
    max_pressurehase_saturation_change: float | None
    max_allowed_phase_saturation_change: float | None
    message: str | None = None


def _validate_pressure_range(
    pressure_grid: NDimensionalGrid[ThreeDimensions],
    time_step: int,
    time_step_size: float,
    time: float,
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None = None,
) -> StepResult[ThreeDimensions] | None:
    """
    Check for out-of-range pressures and return a failure `StepResult` if found.

    :param pressure_grid: Pressure grid to validate.
    :param time_step: Current time step index.
    :param time_step_size: Size of the current time step (seconds).
    :param time: Total elapsed simulation time (seconds).
    :param fluid_properties: Fluid properties for the current state.
    :param rock_properties: Rock properties for the current state.
    :param hysteresis_state: Saturation history for the current state.
    :return: `StepResult` with failure if pressures are out of range, *None* otherwise.
    """
    min_allowable = c.MINIMUM_VALID_PRESSURE - 1e-3
    max_allowable = c.MAXIMUM_VALID_PRESSURE + 1e-3
    out_of_range_mask = (pressure_grid < min_allowable) | (pressure_grid > max_allowable)
    out_of_range_indices = np.argwhere(out_of_range_mask)

    if out_of_range_indices.size > 0:
        min_pressure = float(np.min(pressure_grid))
        max_pressure = float(np.max(pressure_grid))
        logger.warning(
            f"Unphysical pressure detected at {out_of_range_indices.size} cells. "
            f"Range: [{min_pressure:.4f}, {max_pressure:.4f}] psi. "
            f"Allowed: [{min_allowable}, {max_allowable}]."
        )
        message = ""
        if min_pressure < min_allowable:
            message += f"Pressure dropped below {min_allowable} psi (Min: {min_pressure:.4f}).\n"
        if max_pressure > max_allowable:
            message += f"Pressure exceeded {max_allowable} psi (Max: {max_pressure:.4f}).\n"
        message += (
            PRESSURE_ERROR_MSG.format(indices=out_of_range_indices.tolist())
            + f"\nAt Time Step {time_step}."
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
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
    max_pressurehase_saturation_change = None
    max_allowed_phase_saturation_change = None

    oil_tol = max(tolerance, 0.005 * max_allowed_oil_saturation_change)
    effective_allowed_oil = max_allowed_oil_saturation_change + oil_tol

    if maximum_oil_saturation_change > effective_allowed_oil:
        violated = True
        max_pressurehase_saturation_change = maximum_oil_saturation_change
        max_allowed_phase_saturation_change = max_allowed_oil_saturation_change
        messages.append(
            f"Oil saturation change {maximum_oil_saturation_change:.9f} exceeded "
            f"maximum allowed {max_allowed_oil_saturation_change:.9f}."
        )

    water_tolerance = max(tolerance, 0.005 * max_allowed_water_saturation_change)
    effective_allowed_water = max_allowed_water_saturation_change + water_tolerance

    if maximum_water_saturation_change > effective_allowed_water:
        violated = True
        if (
            max_pressurehase_saturation_change is None
            or maximum_water_saturation_change > max_pressurehase_saturation_change
        ):
            max_pressurehase_saturation_change = maximum_water_saturation_change
            max_allowed_phase_saturation_change = max_allowed_water_saturation_change

        messages.append(
            f"Water saturation change {maximum_water_saturation_change:.9f} exceeded "
            f"maximum allowed {max_allowed_water_saturation_change:.9f}."
        )

    gas_tolerance = max(tolerance, 0.005 * max_allowed_gas_saturation_change)
    effective_allowed_gas = max_allowed_gas_saturation_change + gas_tolerance
    if maximum_gas_saturation_change > effective_allowed_gas:
        violated = True
        if (
            max_pressurehase_saturation_change is None
            or maximum_gas_saturation_change > max_pressurehase_saturation_change
        ):
            max_pressurehase_saturation_change = maximum_gas_saturation_change
            max_allowed_phase_saturation_change = max_allowed_gas_saturation_change

        messages.append(
            f"Gas saturation change {maximum_gas_saturation_change:.9f} exceeded "
            f"maximum allowed {max_allowed_gas_saturation_change:.9f}."
        )

    return SaturationChangeCheckResult(
        violated=violated,
        max_pressurehase_saturation_change=max_pressurehase_saturation_change,
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
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    config: Config,
) -> tuple[
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
        hysteresis_state=hysteresis_state,
        disable_capillary_effects=config.disable_capillary_effects,
        capillary_strength_factor=config.capillary_strength_factor,
        phase_appearance_tolerance=config.phase_appearance_tolerance,
    )


def _run_impes_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    elevation_grid: NDimensionalGrid[ThreeDimensions],
    pore_volume_grid: NDimensionalGrid[ThreeDimensions],
    initial_pressure_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    relperm_grids: RelPermGrids[ThreeDimensions],
    relperm_endpoints: RelPermEndpoints,
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
    Execute one time step using IMPES (Implicit Pressure, Explicit Saturation)
    with semi-implicit well handling.

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
    :param hysteresis_state: Saturation history, or *None* if hysteresis is disabled.
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
            relperm_grids=relperm_grids,
            relative_mobility_grids=relative_mobility_grids,
            relperm_endpoints=relperm_endpoints,
            wells=wells,
            time=time,
            config=config,
            wells_indices=wells_indices,
            dtype=dtype,
        )

    else:
        well_rates = None

    logger.debug("Solving pressure implicitly...")
    solve_pressure = (
        implicit.solve_nonlinear_pressure
        if config.use_nonlinear_pressure_solve
        else implicit.solve_pressure
    )
    kwds = (
        dict(
            relperm_grids=relperm_grids,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
            rates=well_rates if has_open_wells else None,
            wells_indices=wells_indices,
            relperm_endpoints=relperm_endpoints,
        )
        if config.use_nonlinear_pressure_solve
        else dict(rates=well_rates if has_open_wells else None)
    )
    pressure_result = solve_pressure(
        elevation_grid=elevation_grid,
        pore_volume_grid=pore_volume_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        config=config,
        **kwds,  # type: ignore
        dtype=dtype,
    )
    if not pressure_result.success:
        logger.error(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    new_pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = float(pressure_solution.maximum_pressure_change)
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
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
            success=False,
            message=message,
            maximum_pressure_change=maximum_pressure_change,
            timer_context={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    result = _validate_pressure_range(
        pressure_grid=new_pressure_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        hysteresis_state=hysteresis_state,
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

    if has_open_wells:
        logger.debug("Re-computing well rates after pressure solve for consistency...")
        well_rates = compute_well_rates(
            fluid_properties=fluid_properties,
            relperm_grids=relperm_grids,
            relative_mobility_grids=relative_mobility_grids,
            relperm_endpoints=relperm_endpoints,
            wells=wells,
            time=time,
            config=config,
            wells_indices=wells_indices,
            dtype=dtype,
        )

    # Refresh boundary conditions after pressure update so that dynamic BCs (Robin,
    # Carter-Tracy) see the new interior pressures before saturation evolves.
    logger.debug("Refreshing boundary conditions after pressure solve...")
    metadata = attrs.evolve(metadata, fluid_properties=fluid_properties)
    flux_boundaries, pressure_boundaries = boundary_conditions.refresh_boundaries(
        metadata=metadata
    )

    # Copy before transport update
    if hysteresis_state is not None:
        old_water_saturation_grid = fluid_properties.water_saturation_grid.copy()
        old_gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()
    else:
        old_water_saturation_grid = None
        old_gas_saturation_grid = None

    logger.debug("Solving transport explicitly...")
    # Correct pore volume for pressure change using `PVn = PVi * exp(Cr * (Pn - Pi))`
    new_pore_volume_grid = pore_volume_grid * np.exp(
        rock_properties.compressibility * (new_pressure_grid - initial_pressure_grid)
    )
    transport_result = explicit.solve_transport(
        elevation_grid=elevation_grid,
        pore_volume_grid=new_pore_volume_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        fluid_properties=fluid_properties,
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
    maximum_oil_saturation_change = float(transport_solution.maximum_oil_saturation_change)
    maximum_water_saturation_change = float(transport_solution.maximum_water_saturation_change)
    maximum_gas_saturation_change = float(transport_solution.maximum_gas_saturation_change)
    saturation_check = _check_saturation_changes(
        maximum_oil_saturation_change=maximum_oil_saturation_change,
        maximum_water_saturation_change=maximum_water_saturation_change,
        maximum_gas_saturation_change=maximum_gas_saturation_change,
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
        "maximum_saturation_change": saturation_check.max_pressurehase_saturation_change,
        "maximum_allowed_saturation_change": saturation_check.max_allowed_phase_saturation_change,
    }
    if not transport_result.success:
        logger.warning(
            f"Explicit saturation evolution failed at time step {time_step}: \n{transport_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
            success=False,
            message=transport_result.message,
            maximum_pressure_change=maximum_pressure_change,
            maximum_oil_saturation_change=maximum_oil_saturation_change,
            maximum_water_saturation_change=maximum_water_saturation_change,
            maximum_gas_saturation_change=maximum_gas_saturation_change,
            timer_context=timer_context,
        )

    if saturation_check.violated:
        message = (
            f"At time step {time_step}, saturation change limits were violated:\n"
            f"{saturation_check.message}\n"
            f"Oil: {maximum_oil_saturation_change:.6f}, "
            f"Water: {maximum_water_saturation_change:.6f}, "
            f"Gas: {maximum_gas_saturation_change:.6f}."
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
            success=False,
            message=message,
            maximum_pressure_change=maximum_pressure_change,
            maximum_oil_saturation_change=maximum_oil_saturation_change,
            maximum_water_saturation_change=maximum_water_saturation_change,
            maximum_gas_saturation_change=maximum_gas_saturation_change,
            timer_context=timer_context,
        )

    logger.debug("Updating fluid properties with new saturation grids...")
    water_saturation_grid = transport_solution.water_saturation_grid
    oil_saturation_grid = transport_solution.oil_saturation_grid
    gas_saturation_grid = transport_solution.gas_saturation_grid
    water_mass_grid = transport_solution.water_mass_grid
    oil_mass_grid = transport_solution.oil_mass_grid
    free_gas_mass_grid = transport_solution.free_gas_mass_grid
    dissolved_gas_mass_in_oil_grid = transport_solution.dissolved_gas_mass_in_oil_grid
    dissolved_gas_mass_in_water_grid = transport_solution.dissolved_gas_mass_in_water_grid
    solvent_concentration_grid = transport_solution.solvent_concentration_grid

    if np.any((gas_saturation_grid > 1) | (oil_saturation_grid > 1)):
        print(
            f"Gas saturation: {gas_saturation_grid[(gas_saturation_grid > 1) | (oil_saturation_grid > 1)]}, ",
            f"Oil saturation: {oil_saturation_grid[(gas_saturation_grid > 1) | (oil_saturation_grid > 1)]}, ",
            f"Water saturation: {water_saturation_grid[(gas_saturation_grid > 1) | (oil_saturation_grid > 1)]}, ",
            f"Pressure: {new_pressure_grid[(gas_saturation_grid > 1) | (oil_saturation_grid > 1)]}, ",
            f"Free gas mass: {free_gas_mass_grid[(gas_saturation_grid > 1) | (oil_saturation_grid > 1)]}, "
            f"Dissolved gas mass: {dissolved_gas_mass_in_oil_grid[(gas_saturation_grid > 1) | (oil_saturation_grid > 1)]},",
            f"Affected cells: {np.argwhere((gas_saturation_grid > 1) | (oil_saturation_grid > 1))}, ",
            f"Available Capacity: {((1.0 - water_saturation_grid - oil_saturation_grid) * fluid_properties.gas_density_grid * new_pore_volume_grid)[(gas_saturation_grid > 1) | (oil_saturation_grid > 1)]}",
        )

    if solvent_concentration_grid is None:
        fluid_properties = attrs.evolve(
            fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_mass_grid=water_mass_grid,
            oil_mass_grid=oil_mass_grid,
            free_gas_mass_grid=free_gas_mass_grid,
            dissolved_gas_mass_in_oil_grid=dissolved_gas_mass_in_oil_grid,
            dissolved_gas_mass_in_water_grid=dissolved_gas_mass_in_water_grid,
        )
    else:
        fluid_properties = attrs.evolve(
            fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_mass_grid=water_mass_grid,
            oil_mass_grid=oil_mass_grid,
            free_gas_mass_grid=free_gas_mass_grid,
            dissolved_gas_mass_in_oil_grid=dissolved_gas_mass_in_oil_grid,
            dissolved_gas_mass_in_water_grid=dissolved_gas_mass_in_water_grid,
            solvent_concentration_grid=solvent_concentration_grid,
        )

    if config.normalize_saturations:
        normalize_saturations(
            oil_saturation_grid=fluid_properties.oil_saturation_grid,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    if hysteresis_state is not None:
        rock_properties, hysteresis_state = update_residual_saturation_grids(
            rock_properties=rock_properties,
            hysteresis_state=hysteresis_state,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            old_water_saturation_grid=old_water_saturation_grid,  # type: ignore
            old_gas_saturation_grid=old_gas_saturation_grid,  # type: ignore
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )

    material_balance_errors = compute_material_balance_errors(
        current_fluid_properties=fluid_properties,
        previous_fluid_properties=initial_fluid_properties,
    )
    timer_context.update({
        "absolute_oil_mbe": material_balance_errors.absolute_oil_mbe,
        "absolute_water_mbe": material_balance_errors.absolute_water_mbe,
        "absolute_gas_mbe": material_balance_errors.absolute_gas_mbe,
        "total_absolute_mbe": material_balance_errors.total_absolute_mbe,
        "relative_oil_mbe": material_balance_errors.relative_oil_mbe,
        "relative_water_mbe": material_balance_errors.relative_water_mbe,
        "relative_gas_mbe": material_balance_errors.relative_gas_mbe,
        "total_relative_mbe": material_balance_errors.total_relative_mbe,
    })
    logger.debug("Transport solve completed.")
    return StepResult(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        hysteresis_state=hysteresis_state,
        rates=well_rates,
        success=True,
        message=transport_result.message,
        maximum_pressure_change=maximum_pressure_change,
        maximum_oil_saturation_change=maximum_oil_saturation_change,
        maximum_water_saturation_change=maximum_water_saturation_change,
        maximum_gas_saturation_change=maximum_gas_saturation_change,
        material_balance_errors=material_balance_errors,
        timer_context=timer_context,
    )


def _run_sequential_implicit_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    elevation_grid: NDimensionalGrid[ThreeDimensions],
    pore_volume_grid: NDimensionalGrid[ThreeDimensions],
    initial_pressure_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    relperm_grids: RelPermGrids[ThreeDimensions],
    relperm_endpoints: RelPermEndpoints,
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
    saturation transport, allowing larger time steps than IMPES in some cases.

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
    :param hysteresis_state: Saturation history, or *None* if hysteresis is disabled.
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
            relperm_grids=relperm_grids,
            relative_mobility_grids=relative_mobility_grids,
            relperm_endpoints=relperm_endpoints,
            wells=wells,
            time=time,
            config=config,
            wells_indices=wells_indices,
            dtype=dtype,
        )

    else:
        well_rates = None

    logger.debug("Solving pressure implicitly...")
    solve_pressure = (
        implicit.solve_nonlinear_pressure
        if config.use_nonlinear_pressure_solve
        else implicit.solve_pressure
    )
    kwds = (
        dict(
            relperm_grids=relperm_grids,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
            rates=well_rates if has_open_wells else None,
            wells_indices=wells_indices,
            relperm_endpoints=relperm_endpoints,
        )
        if config.use_nonlinear_pressure_solve
        else dict(rates=well_rates if has_open_wells else None)
    )
    pressure_result = solve_pressure(
        elevation_grid=elevation_grid,
        pore_volume_grid=pore_volume_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        config=config,
        **kwds,  # type: ignore
        dtype=dtype,
    )
    if not pressure_result.success:
        logger.warning(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    new_pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = float(pressure_solution.maximum_pressure_change)
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
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
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
        time_step_size=time_step_size,
        time=time,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        hysteresis_state=hysteresis_state,
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

    logger.debug("Updating PVT fluid properties to reflect pressure changes...")
    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    if has_open_wells:
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
        logger.debug("Re-computing well rates after pressure solve for consistency...")
        well_rates = compute_well_rates(
            fluid_properties=fluid_properties,
            relperm_grids=relperm_grids,
            relative_mobility_grids=relative_mobility_grids,
            relperm_endpoints=relperm_endpoints,
            wells=wells,
            time=time,
            config=config,
            wells_indices=wells_indices,
            dtype=dtype,
        )

    # Refresh boundary conditions so the saturation solve sees post-pressure BC values.
    metadata = attrs.evolve(metadata, fluid_properties=fluid_properties)
    flux_boundaries, pressure_boundaries = boundary_conditions.refresh_boundaries(
        metadata=metadata
    )

    # Copy before transport update
    if hysteresis_state is not None:
        old_water_saturation_grid = fluid_properties.water_saturation_grid.copy()
        old_gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()
    else:
        old_water_saturation_grid = None
        old_gas_saturation_grid = None

    logger.debug("Solving transport implicitly (Newton-Raphson)...")
    new_pore_volume_grid = pore_volume_grid * np.exp(
        rock_properties.compressibility * (new_pressure_grid - initial_pressure_grid)
    )
    transport_result = implicit.solve_transport(
        elevation_grid=elevation_grid,
        pore_volume_grid=new_pore_volume_grid,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        face_transmissibilities=face_transmissibilities,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        wells_indices=wells_indices,
        rates=well_rates,
        hysteresis_state=hysteresis_state,
        dtype=dtype,
    )
    transport_solution = transport_result.value
    maximum_oil_saturation_change = float(transport_solution.maximum_oil_saturation_change)
    maximum_water_saturation_change = float(transport_solution.maximum_water_saturation_change)
    maximum_gas_saturation_change = float(transport_solution.maximum_gas_saturation_change)
    saturation_check = _check_saturation_changes(
        maximum_oil_saturation_change=maximum_oil_saturation_change,
        maximum_water_saturation_change=maximum_water_saturation_change,
        maximum_gas_saturation_change=maximum_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
        tolerance=1e-4,
    )
    timer_context = {
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
        "maximum_saturation_change": saturation_check.max_pressurehase_saturation_change,
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
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            hysteresis_state=hysteresis_state,
            success=False,
            message=transport_result.message,
            maximum_pressure_change=maximum_pressure_change,
            maximum_oil_saturation_change=maximum_oil_saturation_change,
            maximum_water_saturation_change=maximum_water_saturation_change,
            maximum_gas_saturation_change=maximum_gas_saturation_change,
            timer_context=timer_context,
        )

    if saturation_check.violated:
        if (
            saturation_check.max_pressurehase_saturation_change
            and saturation_check.max_allowed_phase_saturation_change
        ):
            relative_change = (
                abs(
                    saturation_check.max_allowed_phase_saturation_change
                    - saturation_check.max_pressurehase_saturation_change
                )
                / saturation_check.max_allowed_phase_saturation_change
            )
        else:
            relative_change = float("inf")

        if relative_change > config.saturation_change_violation_tolerance:
            message = (
                f"At time step {time_step}, saturation change limits were violated:\n"
                f"{saturation_check.message}\n"
                f"Oil: {maximum_oil_saturation_change:.6f}, "
                f"Water: {maximum_water_saturation_change:.6f}, "
                f"Gas: {maximum_gas_saturation_change:.6f}."
            )
            logger.warning(message)
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                time_step=time_step,
                time_step_size=time_step_size,
                time=time,
                hysteresis_state=hysteresis_state,
                success=False,
                message=message,
                maximum_pressure_change=maximum_pressure_change,
                maximum_oil_saturation_change=maximum_oil_saturation_change,
                maximum_water_saturation_change=maximum_water_saturation_change,
                maximum_gas_saturation_change=maximum_gas_saturation_change,
                timer_context=timer_context,
            )
        # Accept but signal timer to reduce next step
        # Pass the violation info through timer_context

    logger.debug("Updating fluid properties with new saturation grids...")
    water_saturation_grid = transport_solution.water_saturation_grid
    oil_saturation_grid = transport_solution.oil_saturation_grid
    gas_saturation_grid = transport_solution.gas_saturation_grid
    fluid_properties = attrs.evolve(
        fluid_properties,
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        water_mass_grid=transport_solution.water_mass_grid,
        oil_mass_grid=transport_solution.oil_mass_grid,
        free_gas_mass_grid=transport_solution.free_gas_mass_grid,
        dissolved_gas_mass_in_oil_grid=transport_solution.dissolved_gas_mass_in_oil_grid,
        dissolved_gas_mass_in_water_grid=transport_solution.dissolved_gas_mass_in_water_grid,
    )

    if config.normalize_saturations:
        normalize_saturations(
            oil_saturation_grid=fluid_properties.oil_saturation_grid,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    if hysteresis_state is not None:
        rock_properties, hysteresis_state = update_residual_saturation_grids(
            rock_properties=rock_properties,
            hysteresis_state=hysteresis_state,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            old_water_saturation_grid=old_water_saturation_grid,  # type: ignore
            old_gas_saturation_grid=old_gas_saturation_grid,  # type: ignore
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )

    material_balance_errors = compute_material_balance_errors(
        current_fluid_properties=fluid_properties,
        previous_fluid_properties=initial_fluid_properties,
    )
    timer_context.update({
        "absolute_oil_mbe": material_balance_errors.absolute_oil_mbe,
        "absolute_water_mbe": material_balance_errors.absolute_water_mbe,
        "absolute_gas_mbe": material_balance_errors.absolute_gas_mbe,
        "total_absolute_mbe": material_balance_errors.total_absolute_mbe,
        "relative_oil_mbe": material_balance_errors.relative_oil_mbe,
        "relative_water_mbe": material_balance_errors.relative_water_mbe,
        "relative_gas_mbe": material_balance_errors.relative_gas_mbe,
        "total_relative_mbe": material_balance_errors.total_relative_mbe,
    })
    logger.debug("Sequential implicit step completed.")
    return StepResult(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        hysteresis_state=hysteresis_state,
        rates=well_rates,
        success=True,
        message=transport_result.message,
        maximum_pressure_change=maximum_pressure_change,
        maximum_oil_saturation_change=maximum_oil_saturation_change,
        maximum_water_saturation_change=maximum_water_saturation_change,
        maximum_gas_saturation_change=maximum_gas_saturation_change,
        material_balance_errors=material_balance_errors,
        timer_context=timer_context,
    )


def _run_full_sequential_implicit_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    elevation_grid: NDimensionalGrid[ThreeDimensions],
    pore_volume_grid: NDimensionalGrid[ThreeDimensions],
    initial_pressure_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    relperm_grids: RelPermGrids[ThreeDimensions],
    relperm_endpoints: RelPermEndpoints,
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
    :param hysteresis_state: Saturation history, or *None* if hysteresis is disabled.
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
    picard_tolerance = config.picard_tolerance
    maximum_picard_iterations = config.maximum_picard_iterations
    initial_fluid_properties = fluid_properties
    solve_pressure = (
        implicit.solve_nonlinear_pressure
        if config.use_nonlinear_pressure_solve
        else implicit.solve_pressure
    )

    previous_pressure_grid = fluid_properties.pressure_grid.copy()
    previous_water_saturation_grid = fluid_properties.water_saturation_grid.copy()
    previous_oil_saturation_grid = fluid_properties.oil_saturation_grid.copy()
    previous_gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()

    # Copy at old time before transport update
    if hysteresis_state is not None:
        old_water_saturation_grid = fluid_properties.water_saturation_grid.copy()
        old_gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()
    else:
        old_water_saturation_grid = None
        old_gas_saturation_grid = None

    iter_fluid_properties = fluid_properties
    iter_relative_mobility_grids = relative_mobility_grids
    iter_capillary_pressure_grids = capillary_pressure_grids
    iter_relperm_grids = relperm_grids

    converged = False
    transport_result = None
    transport_solution = None
    maximum_pressure_change = None
    maximum_oil_saturation_change = None
    maximum_water_saturation_change = None
    maximum_gas_saturation_change = None
    well_rates = None
    has_open_wells = False
    timer_context: dict[str, typing.Any] = {}

    logger.debug(
        "Starting outer iteration loop (max %d iterations) at time step %d...",
        maximum_picard_iterations,
        time_step,
    )

    for iteration in range(maximum_picard_iterations):
        logger.debug("Outer iteration %d/%d", iteration + 1, maximum_picard_iterations)

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
                relperm_grids=iter_relperm_grids,
                relative_mobility_grids=iter_relative_mobility_grids,
                relperm_endpoints=relperm_endpoints,
                wells=wells,
                time=time,
                config=config,
                wells_indices=wells_indices,
                dtype=dtype,
            )

        else:
            well_rates = None

        logger.debug("Solving pressure implicitly for outer iteration saturation solve...")
        kwds = (
            dict(
                relperm_grids=iter_relperm_grids,
                wells=wells,
                miscibility_model=miscibility_model,
                pvt_tables=config.pvt_tables,
                freeze_saturation_pressure=config.freeze_saturation_pressure,
                rates=well_rates if has_open_wells else None,
                wells_indices=wells_indices,
                relperm_endpoints=relperm_endpoints,
            )
            if config.use_nonlinear_pressure_solve
            else dict(rates=well_rates if has_open_wells else None)
        )
        pressure_result = solve_pressure(
            pore_volume_grid=pore_volume_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=iter_fluid_properties,
            relative_mobility_grids=iter_relative_mobility_grids,
            capillary_pressure_grids=iter_capillary_pressure_grids,
            face_transmissibilities=face_transmissibilities,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
            config=config,
            **kwds,  # type: ignore
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
                time_step=time_step,
                time_step_size=time_step_size,
                time=time,
                hysteresis_state=hysteresis_state,
                success=False,
                message=pressure_result.message,
            )

        pressure_solution = pressure_result.value
        new_pressure_grid = pressure_solution.pressure_grid
        maximum_pressure_change = float(pressure_solution.maximum_pressure_change)
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
                time_step=time_step,
                time_step_size=time_step_size,
                time=time,
                hysteresis_state=hysteresis_state,
                success=False,
                message=message,
                maximum_pressure_change=maximum_pressure_change,
                timer_context={
                    "maximum_pressure_change": maximum_pressure_change,
                    "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
                },
            )

        result = _validate_pressure_range(
            pressure_grid=new_pressure_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            hysteresis_state=hysteresis_state,
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
        logger.debug("Pressure updated in fluid properties for outer iteration saturation solve.")

        logger.debug("Updating fluid properties to reflect pressure change for outer iteration...")
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
            "Solving transport implicitly (Newton-Raphson) for outer iteration %d/%d...",
            iteration + 1,
            maximum_picard_iterations,
        )
        new_pore_volume_grid = pore_volume_grid * np.exp(
            rock_properties.compressibility * (new_pressure_grid - initial_pressure_grid)
        )
        transport_result = implicit.solve_transport(
            elevation_grid=elevation_grid,
            pore_volume_grid=new_pore_volume_grid,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=iter_fluid_properties,
            face_transmissibilities=face_transmissibilities,
            config=config,
            flux_boundaries=flux_boundaries,
            pressure_boundaries=pressure_boundaries,
            wells_indices=wells_indices,
            rates=well_rates,
            hysteresis_state=hysteresis_state,
            dtype=dtype,
        )
        transport_solution = transport_result.value
        maximum_oil_saturation_change = float(transport_solution.maximum_oil_saturation_change)
        maximum_water_saturation_change = float(transport_solution.maximum_water_saturation_change)
        maximum_gas_saturation_change = float(transport_solution.maximum_gas_saturation_change)
        if not transport_result.success:
            logger.warning(
                f"Implicit saturation solve failed at outer iteration "
                f"{iteration + 1}, time step {time_step}:\n{transport_result.message}"
            )
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                time_step=time_step,
                time_step_size=time_step_size,
                time=time,
                hysteresis_state=hysteresis_state,
                success=False,
                message=transport_result.message,
                maximum_pressure_change=maximum_pressure_change,
                maximum_oil_saturation_change=maximum_oil_saturation_change,
                maximum_water_saturation_change=maximum_water_saturation_change,
                maximum_gas_saturation_change=maximum_gas_saturation_change,
            )

        saturation_check = _check_saturation_changes(
            maximum_oil_saturation_change=maximum_oil_saturation_change,
            maximum_water_saturation_change=maximum_water_saturation_change,
            maximum_gas_saturation_change=maximum_gas_saturation_change,
            max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
            max_allowed_water_saturation_change=config.maximum_water_saturation_change,
            max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
            tolerance=1e-4,
        )
        if saturation_check.violated:
            if (
                saturation_check.max_pressurehase_saturation_change
                and saturation_check.max_allowed_phase_saturation_change
            ):
                relative_change = (
                    abs(
                        saturation_check.max_allowed_phase_saturation_change
                        - saturation_check.max_pressurehase_saturation_change
                    )
                    / saturation_check.max_allowed_phase_saturation_change
                )
            else:
                relative_change = float("inf")

            if relative_change > config.saturation_change_violation_tolerance:
                message = (
                    f"At time step {time_step}, outer iteration {iteration + 1}, "
                    f"saturation change limits were violated:\n{saturation_check.message}\n"
                    f"Oil: {maximum_oil_saturation_change:.6f}, "
                    f"Water: {maximum_water_saturation_change:.6f}, "
                    f"Gas: {maximum_gas_saturation_change:.6f}."
                )
                logger.warning(message)
                return StepResult(
                    fluid_properties=fluid_properties,
                    rock_properties=rock_properties,
                    time_step=time_step,
                    time_step_size=time_step_size,
                    time=time,
                    hysteresis_state=hysteresis_state,
                    success=False,
                    message=message,
                    maximum_pressure_change=maximum_pressure_change,
                    maximum_oil_saturation_change=maximum_oil_saturation_change,
                    maximum_water_saturation_change=maximum_water_saturation_change,
                    maximum_gas_saturation_change=maximum_gas_saturation_change,
                    timer_context={
                        "maximum_saturation_change": saturation_check.max_pressurehase_saturation_change,
                        "maximum_allowed_saturation_change": saturation_check.max_allowed_phase_saturation_change,
                    },
                )

        iter_fluid_properties = attrs.evolve(
            iter_fluid_properties,
            water_saturation_grid=transport_solution.water_saturation_grid,
            oil_saturation_grid=transport_solution.oil_saturation_grid,
            gas_saturation_grid=transport_solution.gas_saturation_grid,
            water_mass_grid=transport_solution.water_mass_grid,
            oil_mass_grid=transport_solution.oil_mass_grid,
            free_gas_mass_grid=transport_solution.free_gas_mass_grid,
            dissolved_gas_mass_in_oil_grid=transport_solution.dissolved_gas_mass_in_oil_grid,
            dissolved_gas_mass_in_water_grid=transport_solution.dissolved_gas_mass_in_water_grid,
        )

        if config.normalize_saturations:
            normalize_saturations(
                oil_saturation_grid=iter_fluid_properties.oil_saturation_grid,
                water_saturation_grid=iter_fluid_properties.water_saturation_grid,
                gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
                saturation_epsilon=saturation_epsilon,
            )

        newton_iterations = transport_solution.newton_iterations
        timer_context = {
            "maximum_pressure_change": maximum_pressure_change,
            "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            "maximum_saturation_change": saturation_check.max_pressurehase_saturation_change,
            "maximum_allowed_saturation_change": saturation_check.max_allowed_phase_saturation_change,
            "newton_iterations": newton_iterations,
        }

        # Picard convergence check
        pressure_delta = new_pressure_grid - previous_pressure_grid
        pressure_rms_reference = max(float(np.sqrt(np.mean(new_pressure_grid**2))), 1.0)
        pressure_rms_change = float(np.sqrt(np.mean(pressure_delta**2))) / pressure_rms_reference

        water_saturation_rms_change = float(
            np.sqrt(
                np.mean(
                    (iter_fluid_properties.water_saturation_grid - previous_water_saturation_grid)
                    ** 2
                )
            )
        )
        oil_saturation_rms_change = float(
            np.sqrt(
                np.mean(
                    (iter_fluid_properties.oil_saturation_grid - previous_oil_saturation_grid) ** 2
                )
            )
        )
        gas_saturation_rms_change = float(
            np.sqrt(
                np.mean(
                    (iter_fluid_properties.gas_saturation_grid - previous_gas_saturation_grid) ** 2
                )
            )
        )
        maximum_saturation_rms_change = max(
            water_saturation_rms_change,
            oil_saturation_rms_change,
            gas_saturation_rms_change,
        )

        logger.debug(
            "Picard outer iteration %d/%d: RMS ΔP/P = %.3e, RMS ΔS = %.3e (tol = %.3e)",
            iteration + 1,
            config.maximum_picard_iterations,
            pressure_rms_change,
            maximum_saturation_rms_change,
            picard_tolerance,
        )

        if (
            pressure_rms_change < picard_tolerance
            and maximum_saturation_rms_change < picard_tolerance
        ):
            logger.debug("Picard outer loop converged after %d iteration(s).", iteration + 1)
            converged = True
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
        ) = _rebuild_rock_fluid_grids(
            fluid_properties=iter_fluid_properties,
            rock_properties=rock_properties,
            hysteresis_state=hysteresis_state,
            config=config,
        )

        previous_pressure_grid = new_pressure_grid.copy()
        previous_water_saturation_grid = iter_fluid_properties.water_saturation_grid.copy()
        previous_oil_saturation_grid = iter_fluid_properties.oil_saturation_grid.copy()
        previous_gas_saturation_grid = iter_fluid_properties.gas_saturation_grid.copy()

    if not converged:
        logger.warning(
            f"Outer iteration did not converge after {maximum_picard_iterations} "
            f"iteration(s) at time step {time_step}. Proceeding with last solution."
        )

    assert transport_result is not None and transport_solution is not None, (
        "Saturation solve must have run at least once."
    )

    if hysteresis_state is not None:
        rock_properties, hysteresis_state = update_residual_saturation_grids(
            rock_properties=rock_properties,
            hysteresis_state=hysteresis_state,
            water_saturation_grid=iter_fluid_properties.water_saturation_grid,
            gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
            old_water_saturation_grid=old_water_saturation_grid,  # type: ignore
            old_gas_saturation_grid=old_gas_saturation_grid,  # type: ignore
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )

    material_balance_errors = compute_material_balance_errors(
        current_fluid_properties=iter_fluid_properties,
        previous_fluid_properties=initial_fluid_properties,
    )
    timer_context.update({
        "absolute_oil_mbe": material_balance_errors.absolute_oil_mbe,
        "absolute_water_mbe": material_balance_errors.absolute_water_mbe,
        "absolute_gas_mbe": material_balance_errors.absolute_gas_mbe,
        "total_absolute_mbe": material_balance_errors.total_absolute_mbe,
        "relative_oil_mbe": material_balance_errors.relative_oil_mbe,
        "relative_water_mbe": material_balance_errors.relative_water_mbe,
        "relative_gas_mbe": material_balance_errors.relative_gas_mbe,
        "total_relative_mbe": material_balance_errors.total_relative_mbe,
    })
    logger.debug("Full sequential implicit step completed.")
    return StepResult(
        fluid_properties=iter_fluid_properties,
        rock_properties=rock_properties,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        hysteresis_state=hysteresis_state,
        rates=well_rates,
        success=True,
        message=transport_result.message,
        maximum_pressure_change=maximum_pressure_change,
        maximum_oil_saturation_change=maximum_oil_saturation_change,
        maximum_water_saturation_change=maximum_water_saturation_change,
        maximum_gas_saturation_change=maximum_gas_saturation_change,
        material_balance_errors=material_balance_errors,
        timer_context=timer_context,
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


StepCallback = typing.Callable[[StepResult[ThreeDimensions]], None]
"""A callback invoked after each simulation step attempt."""


@attrs.mutable
class Run(StoreSerializable):
    """
    Simulation run specification.

    Executes a reservoir simulation on a 3D static reservoir model using the
    provided configuration.

    Example:

    ```python
    from bores import BlackOil, Config, Run

    model = BlackOil.from_file("path/to/3d_model.h5")
    config = Config.from_file("path/to/simulation_config.yaml")

    run = Run(model=model, config=config)
    for state in run:
        process(state)
    ```
    """

    model: BlackOil[ThreeDimensions]
    """The reservoir model to simulate."""

    config: Config
    """Simulation configuration and parameters."""

    name: str | None = None
    """Human-readable name for this run."""

    description: str | None = None
    """Detailed description of the simulation."""

    tags: list[str] = attrs.field(factory=list)
    """Tags for organizing runs."""

    created_at: str | None = attrs.field(factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO timestamp of when this run was created."""

    def __call__(
        self,
        config: Config | None = None,
        *,
        on_step_rejected: StepCallback | None = None,
        on_step_accepted: StepCallback | None = None,
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

    def validate(
        self,
        *,
        correct_inplace: bool = True,
        raise_on_error: bool = True,
        zero_flow_tolerance: float | None = None,
        emit_log: bool = True,
    ) -> ValidationReport:
        """
        Run a comprehensive pre-simulation validation on *model* and *config*
        using `validate`.

        :param correct_inplace: When True, small correctable issues are fixed directly
            on the model arrays. When False, they are reported as warnings or errors.
        :param raise_on_error: When True, a `ModelValidationError` is raised after
            all checks complete if any fatal errors were found.
        :param zero_flow_tolerance: Override the tolerance for the zero-flow equilibrium
            check. When None the value is selected automatically from grid scale.
        :param emit_log: When True, each issue is emitted through `logging`.
        :return: Full `ValidationReport`.
        :raises ModelValidationError: If `raise_on_error=True` and any ERROR-severity
            issues were found.
        """
        return validate(
            self.model,
            self.config,
            correct_inplace=correct_inplace,
            raise_on_error=raise_on_error,
            zero_flow_tolerance=zero_flow_tolerance,
            emit_log=emit_log,
        )

    @classmethod
    def read_files(
        cls,
        model_path: str | PathLike,
        config_path: str | PathLike,
        pvt_tables_path: str | PathLike | None = None,
        pvt_data_path: str | PathLike | None = None,
    ) -> Self:
        """
        Load a `Run` from separate model and configuration files.

        :param model_path: Path to the reservoir model file.
        :param config_path: Path to the simulation configuration file.
        :param pvt_tables_path: Optional path to a dumped `PVTTables` file.
        :param pvt_data_path: Optional path to a dumped `PVTDataSet` file.
        :return: `Run` instance with loaded model and config.
        :raises ValidationError: If the loaded model is not a 3-D `BlackOil`.
        """
        model = BlackOil.from_file(model_path)
        if not isinstance(model, BlackOil) or model.dimensions != 3:
            raise ValidationError("Loaded model must be a 3D `BlackOil` instance.")

        config = Config.from_file(config_path)
        if config is None:
            raise ValidationError("Failed to load simulation config from file.")

        if pvt_tables_path is not None:
            pvt_tables = PVTTables.from_file(pvt_tables_path)
            if pvt_tables is None:
                raise ValidationError("Failed to load `PVTTables` data from file.")
            config = config.new(pvt_tables=pvt_tables)

        if pvt_data_path is not None:
            pvt_dataset = PVTDataSet.from_file(pvt_data_path)
            if pvt_dataset is None:
                raise ValidationError("Failed to load `PVTDataSet` from file.")
            pvt_tables = PVTTables.from_dataset(pvt_dataset)
            config = config.new(pvt_tables=pvt_tables)

        return cls(model=model, config=config)


_SCHEME_ALIASES = {
    "si": "Sequential Implicit",
    "full-si": "Full Sequential Implicit",
    "impes": "IMPES",
}


__simulation_running = ContextFlag("simulation_running", initial=False)


def running() -> bool:
    """
    Check if a simulation is currently running in this context.
    i.e Are we in a running simulation?
    """
    return __simulation_running.get()


def stop(reason: str | None = None):
    """
    Stop the currently running simulation (in this context) for the given `reason`

    :param reason: Optional explanation for why the simulation is being stopped.
    """
    if not running():
        return

    if reason is None:
        reason = "Stopped by user."
    raise StopSimulation(reason)


def run(
    input: BlackOil[ThreeDimensions] | Run,
    config: Config | None = None,
    *,
    on_step_rejected: StepCallback | None = None,
    on_step_accepted: StepCallback | None = None,
) -> typing.Generator[ModelState[ThreeDimensions], None, None]:
    """
    Run a simulation on a 3D reservoir model.

    The simulation evolves pressure and saturation over time using the chosen
    evolution scheme.

    :param input: Either a `BlackOil` or a `Run` instance. If a `Run` is
        supplied and *config* is also provided, the explicit *config* takes precedence.
    :param config: Simulation configuration. Required when *input* is a
        `BlackOil`; optional when *input* is a `Run`.
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

    model = bores.BlackOil.from_file("path/to/3d_model.h5")
    config = bores.Config.from_file("path/to/simulation_config.yaml")
    for state in bores.run(model, config):
        process(state)
    ```
    """
    __simulation_running.set(True)
    with ExitStack() as stack:
        # Should always reset the flag when exiting, even on error
        stack.callback(lambda: __simulation_running.reset())
        if isinstance(input, Run):
            model = input.model
            if config is not None:
                logger.info(
                    "Overriding `config` from `Run` instance with the provided `config` parameter."
                )
            config = config or input.config
        else:
            if config is None:
                raise ValidationError("Must provide `config` when `input` is a `BlackOil`.")
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
        log_interval = config.log_interval
        capture_timer_state = config.capture_timer_state
        enable_hysteresis = config.enable_hysteresis
        apply_dip = not config.disable_structural_dip
        needs_injector_seeding = (
            config.minimum_injector_water_saturation or config.minimum_injector_gas_saturation
        )
        relperm_endpoints = (
            config.rock_fluid_tables.relative_permeability_table.get_relperm_endpoints()
        )
        pore_volume_grid = model.pore_volume_grid

        logger.info("Starting simulation workflow...")
        logger.info(
            f"Grid dimensions: (nx={grid_shape[0]}, ny={grid_shape[1]}, nz={grid_shape[2]})"
        )
        logger.info(f"Cell dimensions: (dx={cell_dimension[0]}ft, dy={cell_dimension[1]}ft)")
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

        # Enter constants context
        stack.enter_context(config.constants())
        min_valid_pressure = c.MINIMUM_VALID_PRESSURE
        max_valid_pressure = c.MAXIMUM_VALID_PRESSURE
        saturation_epsilon = c.SATURATION_EPSILON

        fluid_properties = model.fluid_properties
        rock_properties = model.rock_properties
        hysteresis_state = model.hysteresis_state if enable_hysteresis else None
        thickness_grid = model.thickness_grid
        absolute_permeability = rock_properties.absolute_permeability
        net_to_gross_grid = rock_properties.net_to_gross_grid
        elevation_grid = model.build_elevation_grid(apply_dip=apply_dip)
        initial_pressure_grid = fluid_properties.pressure_grid.copy()

        logger.debug("Building well indices cache")
        wells_indices = build_wells_indices(
            grid_shape=grid_shape,
            cell_size_x=cell_dimension[0],
            cell_size_y=cell_dimension[1],
            thickness_grid=thickness_grid,
            wells=wells,
            absolute_permeability=absolute_permeability,
            net_to_gross_grid=net_to_gross_grid,
            regime_constant=-0.75,  # Pseudo steady regime
        )

        logger.debug("Building face transmissibilities...")
        face_transmissibilities = model.build_face_transmissibilities(dtype=dtype)

        # Seed injector saturations to avoid phase deadlock at t=0
        if has_wells and needs_injector_seeding:
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

        logger.debug("Building initial rock-fluid property grids...")
        relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
            _rebuild_rock_fluid_grids(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                hysteresis_state=hysteresis_state,
                config=config,
            )
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
            logger.debug("Attempting time step %d with size %d seconds...", new_step, step_size)
            try:
                if has_wells and has_well_schedules:
                    logger.debug("Updating wells configuration for time step %d", new_step)
                    assert well_schedules is not None
                    well_schedules.apply(wells, state)

                if new_step > 1:
                    if has_wells and needs_injector_seeding:
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
                        # Rebuild rock-fluid grids
                        logger.debug(
                            "Rebuilding rock-fluid property grids after enforcing minimum injector saturations for time step %d...",
                            new_step,
                        )
                        (
                            relperm_grids,
                            relative_mobility_grids,
                            capillary_pressure_grids,
                        ) = _rebuild_rock_fluid_grids(
                            fluid_properties=fluid_properties,
                            rock_properties=rock_properties,
                            hysteresis_state=hysteresis_state,
                            config=config,
                        )

                    if update_wells_indices.get():
                        logger.debug("Updating well indices cache")
                        wells_indices = build_wells_indices(
                            grid_shape=grid_shape,
                            cell_size_x=cell_dimension[0],
                            cell_size_y=cell_dimension[1],
                            thickness_grid=thickness_grid,
                            wells=wells,
                            absolute_permeability=absolute_permeability,
                            net_to_gross_grid=net_to_gross_grid,
                            regime_constant=-0.75,
                        )
                        update_wells_indices.reset()

                kwds = dict(
                    time_step=new_step,
                    grid_shape=grid_shape,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    pore_volume_grid=pore_volume_grid,
                    initial_pressure_grid=initial_pressure_grid,
                    time_step_size=step_size,
                    time=time,
                    face_transmissibilities=face_transmissibilities,
                    rock_properties=rock_properties,
                    fluid_properties=fluid_properties,
                    hysteresis_state=hysteresis_state,
                    relperm_grids=relperm_grids,
                    relperm_endpoints=relperm_endpoints,
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
                    result = _run_impes_step(**kwds)  # type: ignore
                elif scheme in {"sequential-implicit", "si"}:
                    result = _run_sequential_implicit_step(**kwds)  # type: ignore
                elif scheme in {"full-sequential-implicit", "full-si"}:
                    result = _run_full_sequential_implicit_step(**kwds)  # type: ignore
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
                            on_step_accepted(result)

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
                        on_step_rejected(result)

                    continue

                fluid_properties = result.fluid_properties
                rock_properties = result.rock_properties
                hysteresis_state = result.hysteresis_state

                # Update rock-fluid grids using the new pressure and saturation state
                logger.debug("Updating rock-fluid property grids after time step %d...", new_step)
                relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
                    _rebuild_rock_fluid_grids(
                        fluid_properties=fluid_properties,
                        rock_properties=rock_properties,
                        hysteresis_state=hysteresis_state,
                        config=config,
                    )
                )

                if timer.step == 1 or (timer.step % output_frequency == 0) or timer.is_last_step:
                    logger.debug("Capturing model state at time step %d", timer.step)
                    wells_snapshot = copy.deepcopy(wells)
                    if hysteresis_state is not None:
                        model_snapshot = model.evolve(
                            fluid_properties=fluid_properties,
                            rock_properties=rock_properties,
                            hysteresis_state=hysteresis_state,
                        )
                    else:
                        model_snapshot = model.evolve(
                            fluid_properties=fluid_properties,
                            rock_properties=rock_properties,
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
                        relative_mobilities=relative_mobility_grids,
                        relative_permeabilities=relperm_grids,
                        capillary_pressures=capillary_pressure_grids,
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
