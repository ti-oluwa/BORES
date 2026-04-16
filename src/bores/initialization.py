"""Simulation initialization utilities for reservoir simulation cold-start stability."""

import logging
import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.config import Config
from bores.constants import c
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.models import FluidProperties, RockProperties
from bores.solvers.explicit.saturation.immiscible import compute_net_flux_contributions
from bores.transmissibility import FaceTransmissibilities
from bores.types import FluidPhase, ThreeDimensionalGrid, ThreeDimensions
from bores.wells.base import Wells
from bores.wells.indices import WellIndicesCache

logger = logging.getLogger(__name__)


@attrs.frozen
class ZeroFlowViolation:
    """
    Describes a single cell whose net flux at t=0 (before wells open) exceeds
    the configured tolerance.
    """

    cell: typing.Tuple[int, int, int]
    """`(i, j, k)` index of the offending cell."""
    net_water_flux: float
    """Net water flux into the cell (ft³/day). Positive = net inflow."""
    net_oil_flux: float
    """Net oil flux into the cell (ft³/day)."""
    net_gas_flux: float
    """Net gas flux into the cell (ft³/day)."""
    net_total_flux: float
    """`|net_water_flux| + |net_oil_flux| + |net_gas_flux|` (ft³/day)."""
    pore_volume: float
    """Cell pore volume (ft³). Used to normalise the flux for comparison against the relative tolerance."""
    relative_flux: float
    """
    `net_total_flux / pore_volume` (day⁻¹). This is the quantity compared against
    `relative_tolerance` in `check_zero_flow_initialization`.
    """


@attrs.frozen
class ZeroFlowCheckResult:
    """Aggregated result from zero-flow initialization check across all cells."""

    passed: bool
    """`True` when every cell's relative flux is below `relative_tolerance`."""
    max_relative_flux: float
    """Maximum `|net_flux| / pore_volume` observed across all cells (day⁻¹)."""
    max_absolute_flux: float
    """Maximum `sum(|phase_fluxes|)` across all cells (ft³/day)."""
    worst_cell: typing.Optional[typing.Tuple[int, int, int]]
    """`(i, j, k)` index of the cell with the highest relative flux, or `None` when the grid is empty."""
    violations: typing.List[ZeroFlowViolation]
    """All cells (up to `max_reported_violations`) whose relative flux exceeds `relative_tolerance`,
    sorted descending by `relative_flux`."""
    relative_tolerance: float
    """The tolerance that was applied (day⁻¹), for reference."""
    n_cells_checked: int
    """Total number of active cells evaluated."""


def seed_injection_saturations(
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    rock_fluid_tables_config: Config,
    minimum_injector_water_saturation: typing.Optional[float] = None,
    minimum_injector_gas_saturation: typing.Optional[float] = None,
    inplace: bool = False,
) -> FluidProperties[ThreeDimensions]:
    """
    Seed the injected-phase saturation in every injector perforation cell
    so that relative permeability is non-zero before the first pressure solve.

    **Background**

    At t = 0 a water injector perforated in a cell at connate-water saturation
    Swi finds `krw(Swi) = 0` (by definition of the irreducible limit).
    Consequently the phase mobility `λw = krw/μw = 0`, the well productivity
    index `PI_w = WI · λw = 0`, and the BHP-driven injection rate
    `qw = PI_w · (P_cell - BHP) = 0` - regardless of how large the BHP
    constraint is. The saturation transport solver then has no source term for
    water, so Sw stays at Swi on the next step, and the deadlock persists
    indefinitely.

    The same deadlock applies to gas injection when Sg_initial = 0.

    This function resolves the deadlock by adding a small "seed" delta to the
    injected-phase saturation in every perforated cell, backed out of oil
    saturation (which is always the displaced phase for both water and gas
    injection). The seed value is taken from the `minimum_injector_*`
    parameters; if those are `None` the value from `rock_fluid_tables_config`
    is used as a fallback.

    The caller must rebuild relative-mobility grids from the returned
    `FluidProperties` *before* calling the pressure solver.

    :param fluid_properties: Current fluid properties at t = 0 (before any time-stepping).
    :param wells: Wells configuration containing all injection wells.
    :param well_indices_cache: Pre-built cache mapping well names to their perforated cell indices.
        Built by `build_well_indices_cache` in `simulate.py`.
    :param rock_fluid_tables_config: The simulation `Config` instance. Used to read
        `minimum_injector_water_saturation` and `minimum_injector_gas_saturation` when the
        explicit overrides are `None`.
    :param minimum_injector_water_saturation: Override for the minimum water saturation to seed in
        water-injector perforations. When `None`, falls back to
        `rock_fluid_tables_config.minimum_injector_water_saturation`. Must be >
        `phase_appearance_tolerance` to guarantee `krw > 0`.
    :param minimum_injector_gas_saturation: Override for the minimum gas saturation to seed in
        gas-injector perforations. When `None`, falls back to
        `rock_fluid_tables_config.minimum_injector_gas_saturation`. Must be >
        `phase_appearance_tolerance` to guarantee `krg > 0`.
    :param inplace: When `True` the saturation arrays inside `fluid_properties` are
        modified in-place and the *same* `FluidProperties` object is returned. When `False`
        (default) copies are made and the original is left unchanged.
    :return: Updated fluid properties with seeded saturations. Oil saturation has been reduced by
        the seed delta in each affected cell so that `Sw + So + Sg = 1` is preserved everywhere.
    :raises ValueError: If applying the seed would drive oil saturation negative in any cell.
        This means the seed value is larger than the available oil saturation, which indicates
        either an unreasonably large seed or a degenerate initial condition.

    **Notes**

    - Only cells where the current injected-phase saturation is *below* the seed target are
      modified; cells that already satisfy the constraint are left untouched.
    - Oil is always chosen as the displaced phase (the phase whose saturation is reduced) because:
      - Water injection displaces oil, not residual gas.
      - Gas injection also displaces oil in the drainage sense.
    - The function does **not** rebuild relative-permeability or mobility grids.
      That must be done by the caller after this function returns.
    """
    # Resolve effective seed values from config fallbacks
    effective_water_seed: typing.Optional[float] = (
        minimum_injector_water_saturation
        if minimum_injector_water_saturation is not None
        else rock_fluid_tables_config.minimum_injector_water_saturation
    )
    effective_gas_seed: typing.Optional[float] = (
        minimum_injector_gas_saturation
        if minimum_injector_gas_saturation is not None
        else rock_fluid_tables_config.minimum_injector_gas_saturation
    )

    if effective_water_seed is None and effective_gas_seed is None:
        logger.debug("No seed values configured, skipping.")
        return fluid_properties

    if inplace:
        sw = fluid_properties.water_saturation_grid
        so = fluid_properties.oil_saturation_grid
        sg = fluid_properties.gas_saturation_grid
    else:
        sw = fluid_properties.water_saturation_grid.copy()
        so = fluid_properties.oil_saturation_grid.copy()
        sg = fluid_properties.gas_saturation_grid.copy()

    n_water_seeded = 0
    n_gas_seeded = 0

    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_phase = typing.cast(FluidPhase, well.injected_fluid.phase)
        if injected_phase == FluidPhase.WATER:
            seed_value = effective_water_seed
            phase_label = "water"
        else:
            seed_value = effective_gas_seed
            phase_label = "gas"

        if seed_value is None:
            logger.debug(
                "No seed configured for %s injector %r, skipping.",
                phase_label,
                well.name,
            )
            continue

        if well.name not in well_indices_cache.injection:
            logger.warning(
                "Well %r not found in well_indices_cache, skipping.",
                well.name,
            )
            continue

        well_indices = well_indices_cache.injection[well.name]
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell

            if injected_phase == FluidPhase.WATER:
                current_phase_saturation = float(sw[i, j, k])
                if current_phase_saturation >= seed_value:
                    # Already satisfies the constraint
                    continue
                delta = seed_value - current_phase_saturation
                current_oil_saturation = float(so[i, j, k])
                if current_oil_saturation - delta < 0.0:
                    raise ValueError(
                        f"Cannot seed water saturation "
                        f"at cell ({i}, {j}, {k}) for well {well.name!r}. "
                        f"So = {current_oil_saturation:.6f} is insufficient to absorb "
                        f"delta = {delta:.6f} (seed = {seed_value:.6f}, "
                        f"current Sw = {current_phase_saturation:.6f}). "
                        f"Reduce minimum_injector_water_saturation or review "
                        f"the initial saturation distribution."
                    )
                sw[i, j, k] = seed_value
                so[i, j, k] = current_oil_saturation - delta
                n_water_seeded += 1

            else:  # GAS
                current_phase_saturation = float(sg[i, j, k])
                if current_phase_saturation >= seed_value:
                    continue
                delta = seed_value - current_phase_saturation
                current_oil_saturation = float(so[i, j, k])
                if current_oil_saturation - delta < 0.0:
                    raise ValueError(
                        f"Cannot seed gas saturation "
                        f"at cell ({i}, {j}, {k}) for well {well.name!r}. "
                        f"So = {current_oil_saturation:.6f} is insufficient to absorb "
                        f"delta = {delta:.6f} (seed = {seed_value:.6f}, "
                        f"current Sg = {current_phase_saturation:.6f}). "
                        f"Reduce minimum_injector_gas_saturation or review "
                        f"the initial saturation distribution."
                    )
                sg[i, j, k] = seed_value
                so[i, j, k] = current_oil_saturation - delta
                n_gas_seeded += 1

    if n_water_seeded > 0 or n_gas_seeded > 0:
        logger.info(
            "Seeded %d water-injector and %d gas-injector perforations.",
            n_water_seeded,
            n_gas_seeded,
        )

    if inplace:
        return fluid_properties

    return attrs.evolve(
        fluid_properties,
        water_saturation_grid=sw,
        oil_saturation_grid=so,
        gas_saturation_grid=sg,
    )


def apply_minimum_injector_saturations(
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    minimum_injector_water_saturation: typing.Optional[float],
    minimum_injector_gas_saturation: typing.Optional[float],
    dtype: npt.DTypeLike,
) -> FluidProperties[ThreeDimensions]:
    """
    Enforce minimum injected-phase saturation in active injector wellblocks
    **before** the pressure solve of each time step.

    **Background**

    Even after the one-time pre-simulation seeding performed by
    `seed_injection_saturations`, later time steps can cause the
    seeded saturation to drop back below the critical threshold - for example
    if the explicit saturation solver clips it to zero, or if the re-dissolution
    flash removes free gas. Without re-enforcement the injector may fall back
    into the mobility-deadlock described in `seed_injection_saturations`.

    :param fluid_properties: Fluid properties at the start of the current time step.
    :param wells: Wells configuration. Only **open** injection wells with a configured
        `injected_fluid` are processed.
    :param well_indices_cache: Cache of well indices. Only the `injection` sub-cache is used.
    :param minimum_injector_water_saturation: Minimum water saturation to enforce in every active
        water-injector wellblock. Must be > `phase_appearance_tolerance` (see `Config`) to
        guarantee `krw > 0`. Pass `None` to skip water enforcement.
    :param minimum_injector_gas_saturation: Minimum gas saturation to enforce in every active
        gas-injector wellblock. Must be > `phase_appearance_tolerance` to guarantee `krg > 0`.
        Pass `None` to skip gas enforcement.
    :param dtype: NumPy dtype used for all saturation arrays (e.g. `np.float64`).
    :return: New `FluidProperties` instance with updated saturation grids. The original is never
        modified. Oil saturation is reduced by exactly the delta applied to the injected-phase
        saturation in each cell, so that `Sw + So + Sg = 1` is preserved exactly. Cells where
        the current saturation already meets the minimum are left untouched.

    **Notes**

    - Saturation conservation (`Sw + So + Sg = 1`) is maintained by subtracting the seed delta
      from `So`. Oil is chosen as the displaced phase for the same reasons as in
      `seed_injection_saturations`.
    - If `So` would go negative after the subtraction it is clamped to zero and the residual gap
      is silently accepted. A warning is emitted in that case. Callers that want strict
      conservation can normalise saturations afterwards.
    - Only open wells with non-`None` `injected_fluid` are processed. Shut-in injectors
      are skipped.
    - This function does **not** rebuild relative-permeability or mobility grids. That must be
      done by the caller immediately after it returns and before the pressure solve.
    """
    if (
        minimum_injector_water_saturation is None
        and minimum_injector_gas_saturation is None
    ):
        return fluid_properties

    sw: ThreeDimensionalGrid = fluid_properties.water_saturation_grid.copy()
    so: ThreeDimensionalGrid = fluid_properties.oil_saturation_grid.copy()
    sg: ThreeDimensionalGrid = fluid_properties.gas_saturation_grid.copy()

    n_water_enforced = 0
    n_gas_enforced = 0

    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_phase = typing.cast(FluidPhase, well.injected_fluid.phase)
        if injected_phase == FluidPhase.WATER:
            seed = minimum_injector_water_saturation
        else:
            seed = minimum_injector_gas_saturation

        if seed is None:
            continue

        if well.name not in well_indices_cache.injection:
            continue

        for perforation_index in well_indices_cache.injection[well.name]:
            i, j, k = perforation_index.cell

            if injected_phase == FluidPhase.WATER:
                current = float(sw[i, j, k])
                if current >= seed:
                    continue
                delta = dtype(seed - current)  # type: ignore[union-attr]
                oil_available = float(so[i, j, k])
                if oil_available - delta < 0.0:
                    logger.warning(
                        "Oil saturation "
                        "So=%.6f at cell (%d,%d,%d) insufficient to absorb "
                        "water delta=%.6f; clamping So to 0.",
                        oil_available,
                        i,
                        j,
                        k,
                        delta,
                    )
                    delta = dtype(oil_available)  # type: ignore[union-attr]
                sw[i, j, k] = dtype(current + delta)  # type: ignore[union-attr]
                so[i, j, k] = max(dtype(0.0), dtype(oil_available - delta))  # type: ignore[union-attr]
                n_water_enforced += 1

            else:  # GAS
                current = float(sg[i, j, k])
                if current >= seed:
                    continue
                delta = dtype(seed - current)  # type: ignore[union-attr]
                oil_available = float(so[i, j, k])
                if oil_available - delta < 0.0:
                    logger.warning(
                        "Oil saturation "
                        "So=%.6f at cell (%d,%d,%d) insufficient to absorb "
                        "gas delta=%.6f; clamping So to 0.",
                        oil_available,
                        i,
                        j,
                        k,
                        delta,
                    )
                    delta = dtype(oil_available)  # type: ignore[union-attr]
                sg[i, j, k] = dtype(current + delta)  # type: ignore[union-attr]
                so[i, j, k] = max(dtype(0.0), dtype(oil_available - delta))  # type: ignore[union-attr]
                n_gas_enforced += 1

    if n_water_enforced == 0 and n_gas_enforced == 0:
        return fluid_properties

    logger.debug(
        "Enforced minimum in %d water and %d gas injector cells.",
        n_water_enforced,
        n_gas_enforced,
    )
    return attrs.evolve(
        fluid_properties,
        water_saturation_grid=sw,
        oil_saturation_grid=so,
        gas_saturation_grid=sg,
    )


def check_zero_flow_initialization(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    elevation_grid: ThreeDimensionalGrid,
    config: Config,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    relative_tolerance: float = 1e-6,
    max_reported_violations: int = 20,
    dtype: npt.DTypeLike = np.float64,
) -> ZeroFlowCheckResult:
    """
    Verify that the initial pressure-saturation state is at gravitational and
    capillary equilibrium by checking that the net volumetric flux into every
    cell is approximately zero **before any wells are opened**.

    **Background**

    A well-initialised reservoir in hydrostatic and capillary equilibrium satisfies:

    ```
    Σ_faces q_α,face ≈ 0   for all cells, all phases α
    ```

    If this condition is violated it means the initial pressure field and the
    initial saturation/density/capillary-pressure field are inconsistent.
    The simulation will then see spurious artificial flux from the very first
    time step, amplified further once injection begins.

    Common causes:

    - Pressure initialised from a single datum without accounting for the
      capillary-pressure offsets between phases.
    - Density grids not recomputed after pressure or saturation grids were
      modified in the factory.
    - Capillary pressure table inconsistent with the saturation contacts used
      to build the saturation grids.
    - Net-to-gross grid changed after pressure/saturation were initialised.

    The check uses **no-flow Neumann boundaries** on all grid faces (all
    `flux_boundaries` entries are zero, no Dirichlet boundaries), so that
    only the interior pressure/saturation gradients contribute.

    :param fluid_properties: Initial fluid properties (t = 0, before any time-stepping).
    :param rock_properties: Rock properties (porosity, permeability, residual saturations).
    :param face_transmissibilities: Geometric face transmissibilities precomputed by
        `build_face_transmissibilities`. Same object used by the solvers.
    :param elevation_grid: Cell elevation grid (ft), built with the same dip/azimuth as used in
        the simulation.
    :param config: Simulation configuration. Used to build the rock-fluid property grids
        (relative permeability, capillary pressure) and for the `capillary_strength_factor`
        and `disable_capillary_effects` flags.
    :param cell_dimension: `(cell_size_x, cell_size_y)` in feet. Used to compute pore volumes
        for the relative-flux normalisation.
    :param thickness_grid: Cell thickness grid (ft). Used to compute pore volumes.
    :param relative_tolerance: Maximum acceptable `|net_flux| / pore_volume` (day⁻¹) for a
        cell to be considered "at rest". Default `1e-6` day⁻¹. A value of `1e-6` means that
        the spurious flux would change the saturation by at most 1e-6 per day, which is
        negligible for typical simulation time scales.
    :param max_reported_violations: Maximum number of violating cells to include in the returned
        `ZeroFlowCheckResult.violations` list. The worst offenders (highest relative flux)
        are reported first.
    :param dtype: NumPy dtype used for internal flux arrays.
    :return: `ZeroFlowCheckResult` with full results. Key fields:
        - `passed` - `True` when all cells pass.
        - `max_relative_flux` - worst-case relative flux (day⁻¹).
        - `violations` - list of `ZeroFlowViolation` for the worst offending cells.

    **Warnings**

    The function logs a `WARNING` for every violation if `len(violations) > 0` and an
    `INFO` message summarising the outcome.

    **Example**

    Typical call pattern in a user script:

    ```python
    from bores.initialization import check_zero_flow_initialization

    result = check_zero_flow_initialization(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        face_transmissibilities=face_transmissibilities,
        elevation_grid=elevation_grid,
        config=config,
        cell_dimension=model.cell_dimension,
        thickness_grid=model.thickness_grid,
        relative_tolerance=1e-5,
    )
    if not result.passed:
        logger.warning(
            "Initial state is NOT at zero-flow equilibrium! "
            "Max relative flux = %.3e day⁻¹ at cell %s. "
            "Check pressure/saturation/density initialisation.",
            result.max_relative_flux,
            result.worst_cell,
        )
        for v in result.violations[:5]:
            logger.warning("  Cell %s: rel_flux=%.3e", v.cell, v.relative_flux)
    ```
    """
    pressure_grid = fluid_properties.pressure_grid
    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Build relative-permeability, mobility, and capillary-pressure grids
    # using the same path as the main simulation loop.
    _, relative_mobility_grids, capillary_pressure_grids = (
        build_rock_fluid_properties_grids(
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
    )

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    # Build no-flow ghost-cell boundary arrays for the flux computation
    padded_shape = (cell_count_x + 2, cell_count_y + 2, cell_count_z + 2)
    pressure_boundaries: ThreeDimensionalGrid = np.full(
        padded_shape, np.nan, dtype=dtype
    )
    flux_boundaries: ThreeDimensionalGrid = np.zeros(padded_shape, dtype=dtype)

    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )
    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    )

    # Compute per-phase net fluxes using the same upwind kernel as the
    # explicit saturation solver - this is the exact flux that would corrupt
    # the first saturation update if it were non-zero.
    net_water_flux, net_oil_flux, net_gas_flux = compute_net_flux_contributions(
        oil_pressure_grid=pressure_grid,
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
        oil_density_grid=fluid_properties.oil_effective_density_grid,
        water_density_grid=fluid_properties.water_density_grid,
        gas_density_grid=fluid_properties.gas_density_grid,
        elevation_grid=elevation_grid,
        gravitational_constant=gravitational_constant,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        dtype=dtype,
    )

    # Compute per-cell pore volumes for relative-flux normalisation
    pore_volume_grid = (
        cell_size_x
        * cell_size_y
        * thickness_grid
        * rock_properties.net_to_gross_grid
        * rock_properties.porosity_grid
    )

    # Total absolute flux per cell (ft³/day)
    net_total_flux = (
        np.abs(net_water_flux) + np.abs(net_oil_flux) + np.abs(net_gas_flux)
    )

    # Safe relative flux - avoid division by zero in zero-porosity cells
    safe_pore_volume_grid = np.where(pore_volume_grid > 0.0, pore_volume_grid, np.inf)
    relative_flux_grid = net_total_flux / safe_pore_volume_grid  # day⁻¹

    # Identify active cells (non-zero porosity)
    active_mask = rock_properties.porosity_grid > 0.0
    n_cells_checked = int(np.sum(active_mask))

    if n_cells_checked == 0:
        logger.warning("No active cells found (all porosity = 0).")
        return ZeroFlowCheckResult(
            passed=True,
            max_relative_flux=0.0,
            max_absolute_flux=0.0,
            worst_cell=None,
            violations=[],
            relative_tolerance=relative_tolerance,
            n_cells_checked=0,
        )

    active_rel_flux = relative_flux_grid[active_mask]
    active_abs_flux = net_total_flux[active_mask]

    max_relative_flux = float(np.max(active_rel_flux))
    max_absolute_flux = float(np.max(active_abs_flux))

    # Locate the worst cell
    flat_worst = int(np.argmax(relative_flux_grid * active_mask.astype(dtype)))
    worst_i, worst_j, worst_k = np.unravel_index(
        flat_worst,
        (cell_count_x, cell_count_y, cell_count_z),
    )
    worst_cell: typing.Tuple[int, int, int] = (int(worst_i), int(worst_j), int(worst_k))

    # Collect violation cells sorted by descending relative flux
    violation_mask = (relative_flux_grid > relative_tolerance) & active_mask
    n_violations = int(np.sum(violation_mask))
    passed = n_violations == 0

    violations: typing.List[ZeroFlowViolation] = []
    if n_violations > 0:
        violation_indices = np.argwhere(violation_mask)
        # Sort by descending relative flux
        violation_rel_fluxes = relative_flux_grid[violation_mask]
        sort_order = np.argsort(violation_rel_fluxes)[::-1]
        sorted_indices = violation_indices[sort_order]

        for idx in sorted_indices[:max_reported_violations]:
            vi, vj, vk = int(idx[0]), int(idx[1]), int(idx[2])
            pv = float(pore_volume_grid[vi, vj, vk])
            nwf = float(net_water_flux[vi, vj, vk])
            nof = float(net_oil_flux[vi, vj, vk])
            ngf = float(net_gas_flux[vi, vj, vk])
            ntf = abs(nwf) + abs(nof) + abs(ngf)
            violations.append(
                ZeroFlowViolation(
                    cell=(vi, vj, vk),
                    net_water_flux=nwf,
                    net_oil_flux=nof,
                    net_gas_flux=ngf,
                    net_total_flux=ntf,
                    pore_volume=pv,
                    relative_flux=float(relative_flux_grid[vi, vj, vk]),
                )
            )

    # Log summary
    if passed:
        logger.info(
            "PASSED. "
            "Max relative flux = %.3e day⁻¹ (tolerance = %.3e day⁻¹). "
            "Checked %d active cells.",
            max_relative_flux,
            relative_tolerance,
            n_cells_checked,
        )
    else:
        logger.warning(
            "FAILED. "
            "%d of %d active cells exceed tolerance %.3e day⁻¹. "
            "Max relative flux = %.3e day⁻¹ at cell %s. "
            "Check pressure/saturation/density/capillary-pressure consistency.",
            n_violations,
            n_cells_checked,
            relative_tolerance,
            max_relative_flux,
            worst_cell,
        )
        for v in violations[:5]:
            logger.warning(
                "  Cell %s: relative_flux=%.3e day⁻¹  "
                "(qw=%.3e, qo=%.3e, qg=%.3e ft³/day, PV=%.3e ft³)",
                v.cell,
                v.relative_flux,
                v.net_water_flux,
                v.net_oil_flux,
                v.net_gas_flux,
                v.pore_volume,
            )

    return ZeroFlowCheckResult(
        passed=passed,
        max_relative_flux=max_relative_flux,
        max_absolute_flux=max_absolute_flux,
        worst_cell=worst_cell,
        violations=violations,
        relative_tolerance=relative_tolerance,
        n_cells_checked=n_cells_checked,
    )
