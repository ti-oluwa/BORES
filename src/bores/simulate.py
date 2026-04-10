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
    PhaseTensorsProxy,
    Rates,
    SparseTensor,
)
from bores.errors import SimulationError, StopSimulation, TimingError, ValidationError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids, RelPermGrids
from bores.grids.pvt import (
    build_three_phase_relative_mobilities_grids,
    build_three_phase_relative_permeabilities_grids,
)
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
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
from bores.states import ModelState
from bores.stores import StoreSerializable
from bores.tables.pvt import PVTDataSet, PVTTables
from bores.transmissibility import FaceTransmissibilities, build_face_transmissibilities
from bores.types import MiscibilityModel, NDimension, NDimensionalGrid, ThreeDimensions
from bores.updates import (
    apply_solution_gas_updates,
    update_fluid_properties,
    update_residual_saturation_grids,
)
from bores.wells.base import Wells
from bores.wells.indices import (
    WellIndicesCache,
    build_well_indices_cache,
    update_well_indices,
)

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
    """Result from executing one time step of the simulation."""

    fluid_properties: FluidProperties[NDimension]
    """Updated fluid properties after the time step."""
    rock_properties: RockProperties[NDimension]
    """Updated rock properties after the time step."""
    saturation_history: typing.Optional[SaturationHistory[NDimension]] = None
    """Updated saturation history after the time step."""
    injection_rates: typing.Optional[Rates[float, NDimension]] = None
    """Phase injection rates during the time step."""
    production_rates: typing.Optional[Rates[float, NDimension]] = None
    """Phase production rates during the time step."""
    injection_fvfs: typing.Optional[FormationVolumeFactors[float, NDimension]] = None
    """Phase injection formation volume factors during the time step."""
    production_fvfs: typing.Optional[FormationVolumeFactors[float, NDimension]] = None
    """Phase production formation volume factors during the time step."""
    injection_bhps: typing.Optional[BottomHolePressures[float, NDimension]] = None
    """Phase injection bottom hole pressures during the time step."""
    production_bhps: typing.Optional[BottomHolePressures[float, NDimension]] = None
    """Phase production bottom hole pressures during the time step."""
    success: bool = True
    """Whether the time step evolution was successful."""
    message: typing.Optional[str] = None
    """Optional message providing additional information about the time step result."""
    material_balance_errors: typing.Optional[MaterialBalanceErrors] = None
    """Material balance errors for this time step. None for rejected steps or first step."""
    timer_kwargs: typing.Dict[str, typing.Any] = attrs.field(factory=dict)
    """Kwargs that should be passed to the simulation timer on accepting or rejecting a step."""


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
            UNPHYSICAL_PRESSURE_ERROR_MSG.format(indices=out_of_range_indices.tolist())
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
    if maximum_oil_saturation_change > max_allowed_oil_saturation_change * (
        1 + oil_tol
    ):
        violated = True
        max_phase_saturation_change = maximum_oil_saturation_change
        max_allowed_phase_saturation_change = max_allowed_oil_saturation_change
        messages.append(
            f"Oil saturation change {maximum_oil_saturation_change:.6f} exceeded "
            f"maximum allowed {max_allowed_oil_saturation_change:.6f}."
        )

    water_tol = max(tolerance, 0.005 * max_allowed_water_saturation_change)
    if maximum_water_saturation_change > max_allowed_water_saturation_change * (
        1 + water_tol
    ):
        violated = True
        if (
            max_phase_saturation_change is None
            or maximum_water_saturation_change > max_phase_saturation_change
        ):
            max_phase_saturation_change = maximum_water_saturation_change
            max_allowed_phase_saturation_change = max_allowed_water_saturation_change
        messages.append(
            f"Water saturation change {maximum_water_saturation_change:.6f} exceeded "
            f"maximum allowed {max_allowed_water_saturation_change:.6f}."
        )

    gas_tol = max(tolerance, 0.005 * max_allowed_gas_saturation_change)
    if maximum_gas_saturation_change > max_allowed_gas_saturation_change * (
        1 + gas_tol
    ):
        violated = True
        if (
            max_phase_saturation_change is None
            or maximum_gas_saturation_change > max_phase_saturation_change
        ):
            max_phase_saturation_change = maximum_gas_saturation_change
            max_allowed_phase_saturation_change = max_allowed_gas_saturation_change
        messages.append(
            f"Gas saturation change {maximum_gas_saturation_change:.6f} exceeded "
            f"maximum allowed {max_allowed_gas_saturation_change:.6f}."
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


def _fvfs_proxy(
    fvfs: FormationVolumeFactors[float, NDimension],
) -> PhaseTensorsProxy[float, NDimension]:
    return PhaseTensorsProxy(oil=fvfs.oil, water=fvfs.water, gas=fvfs.gas)


def _bhps_proxy(
    bhps: BottomHolePressures[float, NDimension],
) -> PhaseTensorsProxy[float, NDimension]:
    return PhaseTensorsProxy(oil=bhps.oil, water=bhps.water, gas=bhps.gas)


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
        relative_mobility_range=config.relative_mobility_range,
        phase_appearance_tolerance=config.phase_appearance_tolerance,
    )


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
    relperm_grids: typing.Any,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    dtype: npt.DTypeLike,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using (semi-implicit) IMPES (Implicit Pressure, Explicit Saturation).

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
    :param well_indices_cache: Cache of well indices for efficient lookup during the pressure solve.
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

    logger.debug("Evolving pressure (implicit)...")
    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_fvfs = _make_fvfs(grid_shape)
    production_fvfs = _make_fvfs(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)

    pressure_result = implicit.evolve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        wells=wells,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_fvfs=_fvfs_proxy(injection_fvfs),
        production_fvfs=_fvfs_proxy(production_fvfs),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
        well_indices_cache=well_indices_cache,
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
    pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change

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
            timer_kwargs={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    pressure_validation_result = _validate_pressure_range(
        pressure_grid=pressure_grid,
        time_step=time_step,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
    )
    if pressure_validation_result is not None:
        return pressure_validation_result

    np.clip(
        pressure_grid,
        min_valid_pressure,
        max_valid_pressure,
        dtype=dtype,
        out=pressure_grid,
    )

    fluid_properties = attrs.evolve(fluid_properties, pressure_grid=pressure_grid)
    logger.debug("Pressure evolution completed.")

    # Refresh boundary maps after pressure update so that dynamic BCs (Robin,
    # Carter-Tracy) see the new interior pressures before saturation evolves.
    logger.debug("Refreshing boundary maps after pressure solve...")
    metadata = attrs.evolve(metadata, fluid_properties=fluid_properties)
    flux_boundaries, pressure_boundaries = (
        boundary_conditions.refresh_dynamic_boundaries(metadata=metadata)
    )

    logger.debug("Updating PVT fluid properties for saturation evolution...")
    old_rs = fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    old_bo = fluid_properties.oil_formation_volume_factor_grid.copy()
    old_rsw = fluid_properties.gas_solubility_in_water_grid.copy()
    old_bw = fluid_properties.water_formation_volume_factor_grid.copy()
    old_so = fluid_properties.oil_saturation_grid.copy()
    old_sg = fluid_properties.gas_saturation_grid.copy()
    old_sw = fluid_properties.water_saturation_grid.copy()

    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )
    fluid_properties = apply_solution_gas_updates(
        fluid_properties=fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_rs,
        old_oil_formation_volume_factor_grid=old_bo,
        old_gas_solubility_in_water_grid=old_rsw,
        old_water_formation_volume_factor_grid=old_bw,
    )

    flash_check = _check_saturation_changes(
        maximum_oil_saturation_change=float(
            np.max(np.abs(fluid_properties.oil_saturation_grid - old_so))
        ),
        maximum_water_saturation_change=float(
            np.max(np.abs(fluid_properties.water_saturation_grid - old_sw))
        ),
        maximum_gas_saturation_change=float(
            np.max(np.abs(fluid_properties.gas_saturation_grid - old_sg))
        ),
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    if flash_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_check.message}"
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_saturation_change": flash_check.max_phase_saturation_change,
                "maximum_allowed_saturation_change": flash_check.max_allowed_phase_saturation_change,
            },
        )

    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Rebuild rock-fluid grids from post-flash saturations so that krg > 0
    # in cells with newly liberated gas before the saturation transport.
    logger.debug("Rebuilding relative permeability and mobility grids...")
    krw, kro, krg = build_three_phase_relative_permeabilities_grids(
        water_saturation_grid=fluid_properties.water_saturation_grid,
        oil_saturation_grid=fluid_properties.oil_saturation_grid,
        gas_saturation_grid=fluid_properties.gas_saturation_grid,
        irreducible_water_saturation_grid=rock_properties.irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=rock_properties.residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=rock_properties.residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=rock_properties.residual_gas_saturation_grid,
        relative_permeability_table=config.rock_fluid_tables.relative_permeability_table,
        phase_appearance_tolerance=config.phase_appearance_tolerance,
    )
    lw, lo, lg = build_three_phase_relative_mobilities_grids(
        oil_relative_permeability_grid=kro,
        water_relative_permeability_grid=krw,
        gas_relative_permeability_grid=krg,
        water_viscosity_grid=fluid_properties.water_viscosity_grid,
        oil_viscosity_grid=fluid_properties.oil_effective_viscosity_grid,
        gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
    )
    lw = config.relative_mobility_range["water"].clip(lw)
    lo = config.relative_mobility_range["oil"].clip(lo)
    lg = config.relative_mobility_range["gas"].clip(lg)
    relative_mobility_grids = RelativeMobilityGrids(
        water_relative_mobility=lw,
        oil_relative_mobility=lo,
        gas_relative_mobility=lg,
    )

    logger.debug("Evolving saturation (explicit)...")
    pressure_change_grid = pressure_grid - old_pressure_grid

    if miscibility_model == "immiscible":
        saturation_result = explicit.evolve_saturation(
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
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            dtype=dtype,
        )
    else:
        saturation_result = explicit.evolve_miscible_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            relative_mobility_grids=relative_mobility_grids,
            capillary_pressure_grids=capillary_pressure_grids,
            wells=wells,
            config=config,
            flux_boundaries=flux_boundaries,
            pressure_boundaries=pressure_boundaries,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            dtype=dtype,
        )

    saturation_solution = saturation_result.value
    sat_check = _check_saturation_changes(
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
        "maximum_saturation_change": sat_check.max_phase_saturation_change,
        "maximum_allowed_saturation_change": sat_check.max_allowed_phase_saturation_change,
    }

    if not saturation_result.success:
        logger.warning(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if sat_check.violated:
        message = (
            f"At time step {time_step}, saturation change limits were violated:\n"
            f"{sat_check.message}\n"
            f"Oil: {saturation_solution.maximum_oil_saturation_change:.6f}, "
            f"Water: {saturation_solution.maximum_water_saturation_change:.6f}, "
            f"Gas: {saturation_solution.maximum_gas_saturation_change:.6f}."
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    logger.debug("Updating fluid properties with new saturation grids...")
    sw = saturation_solution.water_saturation_grid.astype(dtype, copy=False)
    so = saturation_solution.oil_saturation_grid.astype(dtype, copy=False)
    sg = saturation_solution.gas_saturation_grid.astype(dtype, copy=False)
    solvent = saturation_solution.solvent_concentration_grid

    if solvent is None:
        fluid_properties = attrs.evolve(
            fluid_properties,
            water_saturation_grid=sw,
            oil_saturation_grid=so,
            gas_saturation_grid=sg,
        )
    else:
        fluid_properties = attrs.evolve(
            fluid_properties,
            water_saturation_grid=sw,
            oil_saturation_grid=so,
            gas_saturation_grid=sg,
            solvent_concentration_grid=solvent.astype(dtype, copy=False),
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
        injection_rates=injection_rates,
        production_rates=production_rates,
        time_step_size=time_step_size,
    )
    timer_kwargs.update(
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
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_fvfs=injection_fvfs,
        production_fvfs=production_fvfs,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        material_balance_errors=material_balance_errors,
        timer_kwargs=timer_kwargs,
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
    relperm_grids: typing.Any,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
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
    :param well_indices_cache: Cache of well indices for efficient lookup during the pressure solve.
    :param dtype: Data type used for numerical arrays.
    :param min_valid_pressure: Minimum valid pressure (psi). Pressures below this trigger a failure.
    :param max_valid_pressure: Maximum valid pressure (psi). Pressures above this trigger a failure.
    :param saturation_epsilon: Small value to keep saturations strictly between 0 and 1.
    :return: `StepResult` containing updated fluid properties, rock properties, and rates.
    """
    old_pressure_grid = fluid_properties.pressure_grid.copy()
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

    logger.debug("Evolving pressure (implicit)...")
    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_fvfs = _make_fvfs(grid_shape)
    production_fvfs = _make_fvfs(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)

    pressure_result = implicit.evolve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        wells=wells,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        well_indices_cache=well_indices_cache,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_fvfs=_fvfs_proxy(injection_fvfs),
        production_fvfs=_fvfs_proxy(production_fvfs),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
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
    pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change

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
            timer_kwargs={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    pressure_validation_result = _validate_pressure_range(
        pressure_grid=pressure_grid,
        time_step=time_step,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
    )
    if pressure_validation_result is not None:
        return pressure_validation_result

    np.clip(
        pressure_grid,
        min_valid_pressure,
        max_valid_pressure,
        dtype=dtype,
        out=pressure_grid,
    )
    fluid_properties = attrs.evolve(fluid_properties, pressure_grid=pressure_grid)

    # Refresh boundary maps so the saturation solve sees post-pressure BC values.
    metadata = attrs.evolve(metadata, fluid_properties=fluid_properties)
    flux_boundaries, pressure_boundaries = (
        boundary_conditions.refresh_dynamic_boundaries(metadata=metadata)
    )

    old_rs = fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    old_bo = fluid_properties.oil_formation_volume_factor_grid.copy()
    old_rsw = fluid_properties.gas_solubility_in_water_grid.copy()
    old_bw = fluid_properties.water_formation_volume_factor_grid.copy()
    old_so = fluid_properties.oil_saturation_grid.copy()
    old_sg = fluid_properties.gas_saturation_grid.copy()
    old_sw = fluid_properties.water_saturation_grid.copy()

    logger.debug("Updating PVT fluid properties for saturation evolution...")
    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )
    fluid_properties = apply_solution_gas_updates(
        fluid_properties=fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_rs,
        old_oil_formation_volume_factor_grid=old_bo,
        old_gas_solubility_in_water_grid=old_rsw,
        old_water_formation_volume_factor_grid=old_bw,
    )

    flash_check = _check_saturation_changes(
        maximum_oil_saturation_change=float(
            np.max(np.abs(fluid_properties.oil_saturation_grid - old_so))
        ),
        maximum_water_saturation_change=float(
            np.max(np.abs(fluid_properties.water_saturation_grid - old_sw))
        ),
        maximum_gas_saturation_change=float(
            np.max(np.abs(fluid_properties.gas_saturation_grid - old_sg))
        ),
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    if flash_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_check.message}"
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_saturation_change": flash_check.max_phase_saturation_change,
                "maximum_allowed_saturation_change": flash_check.max_allowed_phase_saturation_change,
            },
        )

    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    logger.debug("Evolving saturation (implicit, Newton-Raphson)...")
    pressure_change_grid = pressure_grid - old_pressure_grid

    saturation_result = implicit.evolve_saturation(
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step_size=time_step_size,
        time=time,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        face_transmissibilities=face_transmissibilities,
        config=config,
        well_indices_cache=well_indices_cache,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
        pressure_change_grid=pressure_change_grid,
        dtype=dtype,
    )
    saturation_solution = saturation_result.value
    sat_check = _check_saturation_changes(
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
        "maximum_saturation_change": sat_check.max_phase_saturation_change,
        "maximum_allowed_saturation_change": sat_check.max_allowed_phase_saturation_change,
        "newton_iterations": saturation_solution.newton_iterations,
    }

    if not saturation_result.success:
        logger.warning(
            f"Implicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if sat_check.violated:
        message = (
            f"At time step {time_step}, saturation change limits were violated:\n"
            f"{sat_check.message}\n"
            f"Oil: {saturation_solution.maximum_oil_saturation_change:.6f}, "
            f"Water: {saturation_solution.maximum_water_saturation_change:.6f}, "
            f"Gas: {saturation_solution.maximum_gas_saturation_change:.6f}."
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    logger.debug("Updating fluid properties with new saturation grids...")
    sw = saturation_solution.water_saturation_grid.astype(dtype, copy=False)
    so = saturation_solution.oil_saturation_grid.astype(dtype, copy=False)
    sg = saturation_solution.gas_saturation_grid.astype(dtype, copy=False)

    fluid_properties = attrs.evolve(
        fluid_properties,
        water_saturation_grid=sw,
        oil_saturation_grid=so,
        gas_saturation_grid=sg,
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
        injection_rates=injection_rates,
        production_rates=production_rates,
        time_step_size=time_step_size,
    )
    timer_kwargs.update(
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
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_fvfs=injection_fvfs,
        production_fvfs=production_fvfs,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        material_balance_errors=material_balance_errors,
        timer_kwargs=timer_kwargs,
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
    relperm_grids: typing.Any,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
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
    :param well_indices_cache: Cache of well indices for efficient lookup during the pressure solve.
    :param dtype: Data type used for numerical arrays.
    :param min_valid_pressure: Minimum valid pressure (psi). Pressures below this trigger a failure.
    :param max_valid_pressure: Maximum valid pressure (psi). Pressures above this trigger a failure.
    :param saturation_epsilon: Small value to keep saturations strictly between 0 and 1.
    :return: `StepResult` containing updated fluid properties, rock properties, and rates.
    """
    saturation_tolerance = config.saturation_outer_convergence_tolerance
    pressure_tolerance = config.pressure_outer_convergence_tolerance
    maximum_newton_iterations = config.maximum_newton_iterations
    maximum_outer_iterations = config.maximum_outer_iterations
    initial_fluid_properties = fluid_properties

    logger.debug(
        f"Outer iteration tolerances - "
        f"saturation (absolute): {saturation_tolerance:.2e}, "
        f"pressure (relative): {pressure_tolerance:.2e}"
    )

    previous_pressure_grid = fluid_properties.pressure_grid.copy()
    previous_sw = fluid_properties.water_saturation_grid.copy()
    previous_so = fluid_properties.oil_saturation_grid.copy()
    previous_sg = fluid_properties.gas_saturation_grid.copy()

    iter_fluid_properties = fluid_properties
    iter_relative_mobility_grids = relative_mobility_grids
    iter_capillary_pressure_grids = capillary_pressure_grids
    iter_relperm_grids = relperm_grids

    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_fvfs = _make_fvfs(grid_shape)
    production_fvfs = _make_fvfs(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)

    outer_converged = False
    saturation_result = None
    saturation_solution = None
    sat_check = None
    maximum_pressure_change = 0.0
    final_timer_kwargs: typing.Dict[str, typing.Any] = {}

    logger.debug(
        f"Starting outer iteration loop (max {maximum_outer_iterations} iterations) "
        f"at time step {time_step}..."
    )

    for iteration in range(maximum_outer_iterations):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Outer iteration %d/%d", iteration + 1, maximum_outer_iterations
            )

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

        pressure_result = implicit.evolve_pressure(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            rock_properties=rock_properties,
            fluid_properties=iter_fluid_properties,
            relative_mobility_grids=iter_relative_mobility_grids,
            capillary_pressure_grids=iter_capillary_pressure_grids,
            face_transmissibilities=face_transmissibilities,
            wells=wells,
            config=config,
            well_indices_cache=well_indices_cache,
            flux_boundaries=flux_boundaries,
            pressure_boundaries=pressure_boundaries,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            injection_fvfs=_fvfs_proxy(injection_fvfs),
            production_fvfs=_fvfs_proxy(production_fvfs),
            injection_bhps=_bhps_proxy(injection_bhps),
            production_bhps=_bhps_proxy(production_bhps),
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
        pressure_grid = pressure_solution.pressure_grid
        maximum_pressure_change = pressure_solution.maximum_pressure_change
        maximum_allowed_pressure_change = config.maximum_pressure_change

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
                timer_kwargs={
                    "maximum_pressure_change": maximum_pressure_change,
                    "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
                },
            )

        pressure_validation_result = _validate_pressure_range(
            pressure_grid=pressure_grid,
            time_step=time_step,
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
        )
        if pressure_validation_result is not None:
            return pressure_validation_result

        np.clip(
            pressure_grid,
            min_valid_pressure,
            max_valid_pressure,
            dtype=dtype,
            out=pressure_grid,
        )

        old_rs = iter_fluid_properties.solution_gas_to_oil_ratio_grid.copy()
        old_bo = iter_fluid_properties.oil_formation_volume_factor_grid.copy()
        old_rsw = iter_fluid_properties.gas_solubility_in_water_grid.copy()
        old_bw = iter_fluid_properties.water_formation_volume_factor_grid.copy()
        pre_flash_so = iter_fluid_properties.oil_saturation_grid.copy()
        pre_flash_sg = iter_fluid_properties.gas_saturation_grid.copy()
        pre_flash_sw = iter_fluid_properties.water_saturation_grid.copy()

        iter_fluid_properties = attrs.evolve(
            iter_fluid_properties, pressure_grid=pressure_grid
        )
        iter_fluid_properties = update_fluid_properties(
            fluid_properties=iter_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
        )
        iter_fluid_properties = apply_solution_gas_updates(
            fluid_properties=iter_fluid_properties,
            old_solution_gas_to_oil_ratio_grid=old_rs,
            old_oil_formation_volume_factor_grid=old_bo,
            old_gas_solubility_in_water_grid=old_rsw,
            old_water_formation_volume_factor_grid=old_bw,
        )

        flash_check = _check_saturation_changes(
            maximum_oil_saturation_change=float(
                np.max(np.abs(iter_fluid_properties.oil_saturation_grid - pre_flash_so))
            ),
            maximum_water_saturation_change=float(
                np.max(
                    np.abs(iter_fluid_properties.water_saturation_grid - pre_flash_sw)
                )
            ),
            maximum_gas_saturation_change=float(
                np.max(np.abs(iter_fluid_properties.gas_saturation_grid - pre_flash_sg))
            ),
            max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
            max_allowed_water_saturation_change=config.maximum_water_saturation_change,
            max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
        )
        if flash_check.violated:
            message = (
                f"Solution gas flash at time step {time_step}, outer iteration "
                f"{iteration + 1} violated saturation change limits: {flash_check.message}"
            )
            logger.warning(message)
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                saturation_history=saturation_history,
                success=False,
                message=message,
                timer_kwargs={
                    "maximum_saturation_change": flash_check.max_phase_saturation_change,
                    "maximum_allowed_saturation_change": flash_check.max_allowed_phase_saturation_change,
                },
            )

        iter_fluid_properties = update_fluid_properties(
            fluid_properties=iter_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
        )

        # Refresh boundary maps with updated pressure for the saturation solve.
        metadata = attrs.evolve(metadata, fluid_properties=iter_fluid_properties)
        flux_boundaries, pressure_boundaries = (
            boundary_conditions.refresh_dynamic_boundaries(metadata=metadata)
        )

        pressure_change_grid = pressure_grid - fluid_properties.pressure_grid
        saturation_result = implicit.evolve_saturation(
            grid_shape=grid_shape,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_size,
            time=time,
            rock_properties=rock_properties,
            fluid_properties=iter_fluid_properties,
            face_transmissibilities=face_transmissibilities,
            config=config,
            well_indices_cache=well_indices_cache,
            flux_boundaries=flux_boundaries,
            pressure_boundaries=pressure_boundaries,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            injection_bhps=_bhps_proxy(injection_bhps),
            production_bhps=_bhps_proxy(production_bhps),
            pressure_change_grid=pressure_change_grid,
            dtype=dtype,
        )

        if not saturation_result.success:
            logger.warning(
                f"Implicit saturation solve failed at outer iteration "
                f"{iteration + 1}, time step {time_step}:\n{saturation_result.message}"
            )
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                saturation_history=saturation_history,
                success=False,
                message=saturation_result.message,
            )

        saturation_solution = saturation_result.value
        sat_check = _check_saturation_changes(
            maximum_oil_saturation_change=saturation_solution.maximum_oil_saturation_change,
            maximum_water_saturation_change=saturation_solution.maximum_water_saturation_change,
            maximum_gas_saturation_change=saturation_solution.maximum_gas_saturation_change,
            max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
            max_allowed_water_saturation_change=config.maximum_water_saturation_change,
            max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
        )
        if sat_check.violated:
            message = (
                f"At time step {time_step}, outer iteration {iteration + 1}, "
                f"saturation change limits were violated:\n{sat_check.message}\n"
                f"Oil: {saturation_solution.maximum_oil_saturation_change:.6f}, "
                f"Water: {saturation_solution.maximum_water_saturation_change:.6f}, "
                f"Gas: {saturation_solution.maximum_gas_saturation_change:.6f}."
            )
            logger.warning(message)
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                saturation_history=saturation_history,
                success=False,
                message=message,
                timer_kwargs={
                    "maximum_saturation_change": sat_check.max_phase_saturation_change,
                    "maximum_allowed_saturation_change": sat_check.max_allowed_phase_saturation_change,
                },
            )

        sw = saturation_solution.water_saturation_grid.astype(dtype, copy=False)
        so = saturation_solution.oil_saturation_grid.astype(dtype, copy=False)
        sg = saturation_solution.gas_saturation_grid.astype(dtype, copy=False)

        iter_fluid_properties = attrs.evolve(
            iter_fluid_properties,
            water_saturation_grid=sw,
            oil_saturation_grid=so,
            gas_saturation_grid=sg,
        )

        if config.normalize_saturations:
            normalize_saturations(
                oil_saturation_grid=iter_fluid_properties.oil_saturation_grid,
                water_saturation_grid=iter_fluid_properties.water_saturation_grid,
                gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
                saturation_epsilon=saturation_epsilon,
            )

        newton_iterations = saturation_solution.newton_iterations
        newton_utilization = newton_iterations / maximum_newton_iterations
        final_timer_kwargs = {
            "maximum_pressure_change": maximum_pressure_change,
            "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            "maximum_saturation_change": sat_check.max_phase_saturation_change,
            "maximum_allowed_saturation_change": sat_check.max_allowed_phase_saturation_change,
            "newton_iterations": newton_iterations,
        }

        if newton_utilization < 0.25:
            logger.debug(
                f"Newton converged in {newton_iterations} iterations "
                f"(utilization {newton_utilization:.0%}), skipping further outer iterations."
            )
            outer_converged = True
            break

        total_sat_change_from_bop = max(
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
        if total_sat_change_from_bop < 0.1 * min(
            config.maximum_oil_saturation_change,
            config.maximum_water_saturation_change,
            config.maximum_gas_saturation_change,
        ):
            logger.debug(
                f"Total saturation change from start of timestep {total_sat_change_from_bop:.3e} "
                f"is small, skipping further outer iterations."
            )
            outer_converged = True
            break

        max_outer_sat_change = max(
            float(
                np.max(
                    np.abs(iter_fluid_properties.water_saturation_grid - previous_sw)
                )
            ),
            float(
                np.max(np.abs(iter_fluid_properties.oil_saturation_grid - previous_so))
            ),
            float(
                np.max(np.abs(iter_fluid_properties.gas_saturation_grid - previous_sg))
            ),
        )
        reference_pressure = max(float(np.mean(np.abs(pressure_grid))), 1.0)
        relative_outer_pressure_change = (
            float(np.max(np.abs(pressure_grid - previous_pressure_grid)))
            / reference_pressure
        )

        logger.debug(
            f"Outer iteration {iteration + 1} convergence - "
            f"Δsat (absolute): {max_outer_sat_change:.3e} (atol={saturation_tolerance:.3e}), "
            f"ΔP (relative): {relative_outer_pressure_change:.3e} (rtol={pressure_tolerance:.3e})"
        )

        if (
            max_outer_sat_change < saturation_tolerance
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

        previous_pressure_grid = pressure_grid.copy()
        previous_sw = iter_fluid_properties.water_saturation_grid.copy()
        previous_so = iter_fluid_properties.oil_saturation_grid.copy()
        previous_sg = iter_fluid_properties.gas_saturation_grid.copy()
        injection_rates = _make_rates(grid_shape)
        production_rates = _make_rates(grid_shape)
        injection_fvfs = _make_fvfs(grid_shape)
        production_fvfs = _make_fvfs(grid_shape)
        injection_bhps = _make_bhps(grid_shape)
        production_bhps = _make_bhps(grid_shape)

    if not outer_converged:
        logger.warning(
            f"Outer iteration did not converge after {config.maximum_outer_iterations} "
            f"iteration(s) at time step {time_step}. Proceeding with last solution."
        )

    assert saturation_result is not None and saturation_solution is not None, (
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
        injection_rates=injection_rates,
        production_rates=production_rates,
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
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_fvfs=injection_fvfs,
        production_fvfs=production_fvfs,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        material_balance_errors=material_balance_errors,
        timer_kwargs=final_timer_kwargs,
    )


def _run_explicit_step(
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
    relperm_grids: typing.Any,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    dtype: npt.DTypeLike,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using the fully explicit scheme.

    Both pressure and saturation are advanced explicitly. This is the simplest
    scheme but imposes the most stringent CFL stability constraint on the time
    step size.

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
    :param well_indices_cache: Cache of well indices for efficient lookup during the pressure solve.
    :param dtype: Data type used for numerical arrays.
    :param min_valid_pressure: Minimum valid pressure (psi). Pressures below this trigger a failure.
    :param max_valid_pressure: Maximum valid pressure (psi). Pressures above this trigger a failure.
    :param saturation_epsilon: Small value to keep saturations strictly between 0 and 1.
    :return: `StepResult` containing updated fluid properties, rock properties, and rates.
    """
    old_pressure_grid = fluid_properties.pressure_grid.copy()
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

    logger.debug("Evolving pressure (explicit)...")
    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_fvfs = _make_fvfs(grid_shape)
    production_fvfs = _make_fvfs(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)

    pressure_result = explicit.evolve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        wells=wells,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        well_indices_cache=well_indices_cache,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_fvfs=_fvfs_proxy(injection_fvfs),
        production_fvfs=_fvfs_proxy(production_fvfs),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
        dtype=dtype,
    )
    pressure_solution = pressure_result.value
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change
    timer_kwargs: typing.Dict[str, typing.Any] = {
        "maximum_cfl_encountered": pressure_solution.maximum_cfl_encountered,
        "cfl_threshold": pressure_solution.cfl_threshold,
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
    }

    if not pressure_result.success:
        logger.warning(
            f"Explicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
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
        logger.warning(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    pressure_grid = pressure_solution.pressure_grid
    pressure_validation_result = _validate_pressure_range(
        pressure_grid=pressure_grid,
        time_step=time_step,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
    )
    if pressure_validation_result is not None:
        return StepResult(
            fluid_properties=pressure_validation_result.fluid_properties,
            rock_properties=pressure_validation_result.rock_properties,
            saturation_history=pressure_validation_result.saturation_history,
            success=False,
            message=pressure_validation_result.message,
            timer_kwargs=timer_kwargs,
        )

    np.clip(
        pressure_grid,
        min_valid_pressure,
        max_valid_pressure,
        dtype=dtype,
        out=pressure_grid,
    )
    logger.debug("Pressure evolution completed.")

    # Explicit scheme re-uses current PVT properties for saturation transport;
    # BCs evaluated at start-of-step pressure are appropriate here.
    logger.debug("Evolving saturation (explicit)...")
    pressure_change_grid = pressure_grid - old_pressure_grid

    if miscibility_model == "immiscible":
        saturation_result = explicit.evolve_saturation(
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
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            dtype=dtype,
        )
    else:
        saturation_result = explicit.evolve_miscible_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            relative_mobility_grids=relative_mobility_grids,
            capillary_pressure_grids=capillary_pressure_grids,
            wells=wells,
            config=config,
            flux_boundaries=flux_boundaries,
            pressure_boundaries=pressure_boundaries,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            dtype=dtype,
        )

    saturation_solution = saturation_result.value
    sat_check = _check_saturation_changes(
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
        "maximum_saturation_change": sat_check.max_phase_saturation_change,
        "maximum_allowed_saturation_change": sat_check.max_allowed_phase_saturation_change,
    }

    if not saturation_result.success:
        logger.warning(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if sat_check.violated:
        message = (
            f"At time step {time_step}, saturation change limits were violated:\n"
            f"{sat_check.message}\n"
            f"Oil: {saturation_solution.maximum_oil_saturation_change:.6f}, "
            f"Water: {saturation_solution.maximum_water_saturation_change:.6f}, "
            f"Gas: {saturation_solution.maximum_gas_saturation_change:.6f}."
        )
        logger.warning(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    logger.debug("Saturation evolution completed.")
    sw = saturation_solution.water_saturation_grid.astype(dtype, copy=False)
    so = saturation_solution.oil_saturation_grid.astype(dtype, copy=False)
    sg = saturation_solution.gas_saturation_grid.astype(dtype, copy=False)
    solvent = saturation_solution.solvent_concentration_grid

    if solvent is None:
        fluid_properties = attrs.evolve(
            fluid_properties,
            pressure_grid=pressure_grid,
            water_saturation_grid=sw,
            oil_saturation_grid=so,
            gas_saturation_grid=sg,
        )
    else:
        fluid_properties = attrs.evolve(
            fluid_properties,
            pressure_grid=pressure_grid,
            water_saturation_grid=sw,
            oil_saturation_grid=so,
            gas_saturation_grid=sg,
            solvent_concentration_grid=solvent.astype(dtype, copy=False),
        )

    if config.normalize_saturations:
        normalize_saturations(
            oil_saturation_grid=fluid_properties.oil_saturation_grid,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    logger.debug("Updating PVT fluid properties after explicit solve...")
    old_rs = fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    old_bo = fluid_properties.oil_formation_volume_factor_grid.copy()
    old_rsw = fluid_properties.gas_solubility_in_water_grid.copy()
    old_bw = fluid_properties.water_formation_volume_factor_grid.copy()
    old_so_pre_flash = fluid_properties.oil_saturation_grid.copy()
    old_sg_pre_flash = fluid_properties.gas_saturation_grid.copy()
    old_sw_pre_flash = fluid_properties.water_saturation_grid.copy()

    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )
    fluid_properties = apply_solution_gas_updates(
        fluid_properties=fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_rs,
        old_oil_formation_volume_factor_grid=old_bo,
        old_gas_solubility_in_water_grid=old_rsw,
        old_water_formation_volume_factor_grid=old_bw,
    )

    flash_check = _check_saturation_changes(
        maximum_oil_saturation_change=float(
            np.max(np.abs(fluid_properties.oil_saturation_grid - old_so_pre_flash))
        ),
        maximum_water_saturation_change=float(
            np.max(np.abs(fluid_properties.water_saturation_grid - old_sw_pre_flash))
        ),
        maximum_gas_saturation_change=float(
            np.max(np.abs(fluid_properties.gas_saturation_grid - old_sg_pre_flash))
        ),
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    if flash_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_check.message}"
        )
        logger.warning(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_saturation_change": flash_check.max_phase_saturation_change,
                "maximum_allowed_saturation_change": flash_check.max_allowed_phase_saturation_change,
            },
        )

    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
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
        injection_rates=injection_rates,
        production_rates=production_rates,
        time_step_size=time_step_size,
    )
    timer_kwargs.update(
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
    return StepResult(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_fvfs=injection_fvfs,
        production_fvfs=production_fvfs,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        material_balance_errors=material_balance_errors,
        timer_kwargs=timer_kwargs,
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
    logger.info("Total simulation time: %.1f seconds", timer.simulation_time)
    logger.info("Output frequency: every %d steps", output_frequency)
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

        logger.debug("Building well indices cache")
        well_indices_cache = build_well_indices_cache(
            grid_shape=grid_shape,
            cell_size_x=cell_dimension[0],
            cell_size_y=cell_dimension[1],
            thickness_grid=thickness_grid,
            wells=wells,
            absolute_permeability=rock_properties.absolute_permeability,
            boundary_conditions=boundary_conditions,
        )

        elevation_grid = model.get_elevation_grid(
            apply_dip=not config.disable_structural_dip
        )

        logger.debug("Building face transmissibilities")
        face_transmissibilities = build_face_transmissibilities(
            absolute_permeability=rock_properties.absolute_permeability,
            thickness_grid=thickness_grid,
            cell_size_x=cell_dimension[0],
            cell_size_y=cell_dimension[1],
            dtype=dtype,
        )

        logger.debug("Initializing PVT fluid properties...")
        fluid_properties = update_fluid_properties(
            fluid_properties=fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=pvt_tables,
            freeze_saturation_pressure=freeze_saturation_pressure,
        )
        model = model.evolve(fluid_properties=fluid_properties)

        logger.debug("Building initial rock-fluid property grids...")
        relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
            _rebuild_rock_fluid_grids(fluid_properties, rock_properties, config)
        )

        injection_rates = _make_rates(grid_shape)
        production_rates = _make_rates(grid_shape)
        injection_fvfs = _make_fvfs(grid_shape)
        production_fvfs = _make_fvfs(grid_shape)
        injection_bhps = _make_bhps(grid_shape)
        production_bhps = _make_bhps(grid_shape)
        null_mbe = MaterialBalanceErrors.null()

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
            injection_formation_volume_factors=injection_fvfs,
            production_formation_volume_factors=production_fvfs,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            timer_state=timer.dump_state(),
            material_balance_errors=null_mbe,
        )

        logger.debug("Yielding initial model state")
        yield state

        while not timer.done():
            new_step = timer.next_step
            step_size = timer.propose_step_size()
            time = timer.elapsed_time + step_size
            logger.debug(
                f"Attempting time step {new_step} with size {step_size} seconds..."
            )
            try:
                if has_wells and has_well_schedules:
                    logger.debug(
                        f"Updating wells configuration for time step {new_step}"
                    )
                    well_schedules.apply(wells, state)  # type: ignore[attr-defined]
                    logger.debug("Wells updated.")

                if new_step > 1:
                    # Rebuild rock-fluid grids from the current saturation state
                    # at the start of every step so the solvers see consistent mobilities.
                    logger.debug(
                        f"Rebuilding rock-fluid property grids for time step {new_step}..."
                    )
                    relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
                        _rebuild_rock_fluid_grids(
                            fluid_properties, rock_properties, config
                        )
                    )

                    with update_well_indices as should_update:
                        if should_update:
                            logger.debug("Updating well indices cache")
                            well_indices_cache = build_well_indices_cache(
                                grid_shape=grid_shape,
                                cell_size_x=cell_dimension[0],
                                cell_size_y=cell_dimension[1],
                                thickness_grid=thickness_grid,
                                wells=wells,
                                absolute_permeability=rock_properties.absolute_permeability,
                                boundary_conditions=boundary_conditions,
                            )

                step_kwargs = dict(  # noqa
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
                    well_indices_cache=well_indices_cache,
                    dtype=dtype,
                    min_valid_pressure=min_valid_pressure,
                    max_valid_pressure=max_valid_pressure,
                    saturation_epsilon=saturation_epsilon,
                )

                if scheme == "impes":
                    result = _run_impes_step(**step_kwargs) # type: ignore[arg-type]
                elif scheme in {"sequential-implicit", "si"}:
                    result = _run_sequential_implicit_step(**step_kwargs) # type: ignore[arg-type]
                elif scheme in {"full-sequential-implicit", "full-si"}:
                    result = _run_full_sequential_implicit_step(**step_kwargs) # type: ignore[arg-type]
                elif scheme == "explicit":
                    result = _run_explicit_step(**step_kwargs) # type: ignore[arg-type]
                else:
                    raise ValidationError(
                        f"Invalid simulation scheme {scheme!r}. "
                        "Available schemes: 'impes', 'sequential-implicit', "
                        "'full-sequential-implicit', 'explicit'."
                    )

                acceptable = False
                timer_kwargs = result.timer_kwargs
                error_msg = None
                if result.success:
                    acceptable, error_msg = timer.is_acceptable(**timer_kwargs)
                    if acceptable:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "Time step %d completed successfully.", new_step
                            )
                        timer.accept_step(step_size=step_size, **timer_kwargs)
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
                        f"Time step {new_step} failed with step size {step_size}. "
                        "Retrying with smaller step size."
                    )
                    try:
                        timer.reject_step(
                            step_size=step_size,
                            aggressive=timer.rejection_count > 5,
                            **timer_kwargs,
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Capturing model state at time step %d", timer.step
                        )
                    injection_rates = result.injection_rates
                    production_rates = result.production_rates
                    injection_fvfs = result.injection_fvfs
                    production_fvfs = result.production_fvfs
                    injection_bhps = result.injection_bhps
                    production_bhps = result.production_bhps
                    assert injection_rates is not None
                    assert production_rates is not None
                    assert injection_fvfs is not None
                    assert production_fvfs is not None
                    assert injection_bhps is not None
                    assert production_bhps is not None

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
                    state = ModelState(
                        step=timer.step,
                        step_size=timer.step_size,
                        time=timer.elapsed_time,
                        model=model_snapshot,
                        wells=wells_snapshot,
                        relative_mobilities=mobilities_snapshot,
                        relative_permeabilities=relperm_snapshot,
                        capillary_pressures=capillary_snapshot,
                        injection_rates=injection_rates,
                        production_rates=production_rates.abs(),
                        injection_formation_volume_factors=injection_fvfs,
                        production_formation_volume_factors=production_fvfs,
                        injection_bhps=injection_bhps,
                        production_bhps=production_bhps,
                        timer_state=timer.dump_state() if capture_timer_state else None,
                        material_balance_errors=material_balance_errors
                        if material_balance_errors is not None
                        else null_mbe,
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
