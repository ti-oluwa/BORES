"""
Resolution of many wells in one compiled pass, spread across threads.

`resolve_well_control` resolves one well, in Python, and spends most of its time on Python
around the compiled solvers it calls. `resolve_wells_in_parallel` resolves every well it can take
in a single compiled call that works well by well in parallel, and leaves the rest to
`resolve_well_control`. The two give the same result for a well they both take.
"""

import typing

import numba
import numpy as np

from bores.errors import StopSimulation, ValidationError
from bores.precision import get_dtype
from bores.types import (
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    PhaseValues,
)
from bores.wells.compile import (
    UNSET_INT,
    CompiledLimits,
    CompiledWellSystem,
    InjectorControlModeTag,
    LimitKind,
    ProducerControlModeTag,
    WellKind,
)
from bores.wells.hydraulics.base import (
    ConnectionFluids,
    SurfaceSample,
    WellBoreModel,
    get_hydrostatic_divisor,
)
from bores.wells.hydraulics.beggs_and_brill import BeggsAndBrillWellbore
from bores.wells.hydraulics.gray import GrayWellbore
from bores.wells.hydraulics.hagedorn_brown import HagedornBrownWellbore
from bores.wells.hydraulics.homogeneous import HomogeneousWellbore
from bores.wells.hydraulics.hydrostatic import get_pressure_per_density_and_depth
from bores.wells.hydraulics.tubing_fluids import get_stb_to_volume_factor
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.hydraulics.woldesemayat_ghajar import WoldesemayatGhajarWellbore
from bores.wells.resolution.engine import report_tubing_head_pressure
from bores.wells.resolution.fixed_point import (
    bisect_beggs_and_brill,
    bisect_gray,
    bisect_hagedorn_brown,
    bisect_homogeneous,
    bisect_hydrostatic,
    bisect_uniform,
    bisect_woldesemayat_ghajar,
    solve_beggs_and_brill,
    solve_gray,
    solve_hagedorn_brown,
    solve_homogeneous,
    solve_hydrostatic,
    solve_uniform,
    solve_woldesemayat_ghajar,
)
from bores.wells.resolution.limits import check_economic_violation, find_bhp_bound
from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
from bores.wells.resolution.tubing import WellTubing, resolve_well_tubing
from bores.wells.workspace import (
    ConnectionRates,
    PerforationWorkspace,
    WellsWorkspace,
    sum_connection_rates,
)

__all__ = [
    "CompiledSettings",
    "CompiledWellInputs",
    "CompiledWellOutputs",
    "CompiledWellbores",
    "resolve_wells_in_parallel",
]


NOT_ATTEMPTED = 0
"""`resolutions` entry of a well the compiled pass did not take."""

RESOLVED = 1
"""`resolutions` entry of a well the compiled pass resolved."""

FAILED = 2
"""
`resolutions` entry of a well whose solve did not converge. Resolving it
one at a time gives the error.
"""


PRESSURE_HYDROSTATIC = 1
PRESSURE_UNIFORM = 2

HOMOGENEOUS = 0
BEGGS_AND_BRILL = 1
HAGEDORN_BROWN = 2
GRAY = 3
WOLDESEMAYAT_GHAJAR = 4

CORRELATION_TAGS: dict[str, int] = {
    "homogeneous": HOMOGENEOUS,
    "beggs_brill": BEGGS_AND_BRILL,
    "beggs_and_brill": BEGGS_AND_BRILL,
    "hagedorn_brown": HAGEDORN_BROWN,
    "gray": GRAY,
    "woldesemayat_ghajar": WOLDESEMAYAT_GHAJAR,
}
"""The tag of each hydraulics correlation, by the name of its `WellBoreModel`."""

CORRELATION_CLASSES: tuple[type, ...] = (
    HomogeneousWellbore,
    BeggsAndBrillWellbore,
    HagedornBrownWellbore,
    GrayWellbore,
    WoldesemayatGhajarWellbore,
)
"""The options type of each correlation, indexed by its tag."""

PRODUCER_BHP = int(ProducerControlModeTag.BHP)
PRODUCER_RATE_MODES = (
    int(ProducerControlModeTag.OIL_RATE),
    int(ProducerControlModeTag.WATER_RATE),
    int(ProducerControlModeTag.GAS_RATE),
    int(ProducerControlModeTag.LIQUID_RATE),
    int(ProducerControlModeTag.RESERVOIR_VOLUME_RATE),
)
PRODUCER_RESERVOIR_RATE = int(ProducerControlModeTag.RESERVOIR_VOLUME_RATE)
INJECTOR_BHP = int(InjectorControlModeTag.BHP)
INJECTOR_RATE = int(InjectorControlModeTag.RATE)
INJECTOR_RESERVOIR_RATE = int(InjectorControlModeTag.RESERVOIR_VOLUME_RATE)
INJECTOR_KIND = int(WellKind.INJECTOR)
LIMIT_BHP = int(LimitKind.BHP)
LIMIT_RATE = int(LimitKind.RATE)
LIMIT_ECONOMIC = int(LimitKind.ECONOMIC)
RESERVOIR_QUANTITY = 4
"""`RateQuantityTag.RESERVOIR`."""


class CompiledSettings(typing.NamedTuple):
    """Numbers shared by every well in a compiled pass."""

    pressure_kind: int
    """
    How a well with no wellbore model of its own distributes pressure across its connections:
    `PRESSURE_HYDROSTATIC` or `PRESSURE_UNIFORM`.
    """

    radial_flow_constant: Number
    """Multiplies a well index, a mobility and a drawdown to give a reservoir-condition rate."""

    hydrostatic_scale: Number
    """Pressure of a unit column of unit density, for hydrostatic connection pressures."""

    max_iterations: Integer
    """Iteration limit of the connection pressure and rate solve."""

    tolerance: Number
    """Relative tolerance of that solve."""

    max_bisections: Integer
    """Iteration limit of the search for the pressure that meets a rate target."""

    producer_bhp_floor: Number
    """Lowest bottom-hole pressure a producer is searched down to."""

    injector_bracket_multiplier: Number
    """How far above the highest cell pressure an injector is searched."""

    stb_to_volume_factor: Number
    """Volume of one stock-tank barrel in the volume unit of reservoir rates."""


class CompiledWellbores(typing.NamedTuple):
    """The hydraulics correlation of every well, as arrays a compiled pass can index."""

    tags: IntArray[OneDimension]
    """Per well, the tag of its correlation, or `-1` for a well with none."""

    indices: IntArray[OneDimension]
    """Per well, the position of its options in the array of its own correlation."""

    unit_scales: NumberArray[OneDimension]
    """Per well, the divisor that turns density times gravity times length into a pressure."""

    homogeneous: np.ndarray
    """Options of the wells using the homogeneous correlation, one record each."""

    beggs_and_brill: np.ndarray
    """Options of the wells using Beggs and Brill, one record each."""

    hagedorn_brown: np.ndarray
    """Options of the wells using Hagedorn and Brown, one record each."""

    gray: np.ndarray
    """Options of the wells using Gray, one record each."""

    woldesemayat_ghajar: np.ndarray
    """Options of the wells using Woldesemayat and Ghajar, one record each."""


class CompiledWellInputs(typing.NamedTuple):
    """What a compiled pass reads about every well, as arrays."""

    perforation_offsets: IntArray[OneDimension]
    """Per well, where its connections start in the per-connection arrays. One extra entry."""

    well_indices: NumberArray[OneDimension]
    """Per connection, its well index."""

    representative_depths: NumberArray[OneDimension]
    """Per connection, its true vertical depth."""

    representative_measured_depths: NumberArray[OneDimension]
    """Per connection, its measured depth."""

    inclinations_from_vertical: NumberArray[OneDimension]
    """Per connection, the inclination of the wellbore from vertical there."""

    connection_conductivities: NumberArray[OneDimension]
    """Per connection, the part of its well index that skin and non-Darcy flow do not change."""

    reference_depths: NumberArray[OneDimension]
    """Per well, its bottom-hole pressure datum."""

    reference_measured_depths: NumberArray[OneDimension]
    """Per well, the measured depth of that datum."""

    d_factors: NumberArray[OneDimension]
    """Per well, its non-Darcy factor, `NaN` for none."""

    crossflow_flags: IntArray[OneDimension]
    """Per well, `1` if it allows crossflow."""

    well_kinds: IntArray[OneDimension]
    """Per well, `WellKind`."""

    injected_phases: IntArray[OneDimension]
    """Per well, the phase an injector injects."""

    control_modes: IntArray[OneDimension]
    """Per well, the control mode tag it is resolved under now."""

    target_rates: NumberArray[OneDimension]
    """Per well, the rate target it is resolved under now."""

    target_bhps: NumberArray[OneDimension]
    """Per well, its bottom-hole pressure target."""

    limit_well_offsets: IntArray[OneDimension]
    """Per well, where its limits start in the per-limit arrays. One extra entry."""

    limit_kinds: IntArray[OneDimension]
    """Per limit, `LimitKind`."""

    limit_quantities: IntArray[OneDimension]
    """Per limit, the quantity it applies to."""

    limit_min_values: NumberArray[OneDimension]
    """Per limit, its lower bound."""

    limit_max_values: NumberArray[OneDimension]
    """Per limit, its upper bound."""

    limit_workover_actions: IntArray[OneDimension]
    """Per limit, what to do when it is breached."""

    limit_end_run_flags: IntArray[OneDimension]
    """Per limit, whether breaching it ends the run."""

    fluid_starts: IntArray[OneDimension]
    """Per well, where its connections start in the stacked connection fluids."""


class CompiledWellOutputs(typing.NamedTuple):
    """What a compiled pass writes, which are the arrays of a `WellsWorkspace`."""

    bhps: NumberArray[OneDimension]
    """Per well, its bottom-hole pressure."""

    oil_rates: NumberArray[OneDimension]
    """Per well, its reservoir-condition oil rate."""

    water_rates: NumberArray[OneDimension]
    """Per well, its reservoir-condition water rate."""

    gas_rates: NumberArray[OneDimension]
    """Per well, its reservoir-condition gas rate."""

    surface_oil_rates: NumberArray[OneDimension]
    """Per well, its surface-condition oil rate."""

    surface_water_rates: NumberArray[OneDimension]
    """Per well, its surface-condition water rate."""

    surface_gas_rates: NumberArray[OneDimension]
    """Per well, its surface-condition gas rate."""

    active_limit_rows: IntArray[OneDimension]
    """Per well, the limit row that governs it, or `UNSET_INT`."""

    economic_shutins: IntArray[OneDimension]
    """Per well, `1` if an economic limit shut it in."""

    connection_pressures: NumberArray[OneDimension]
    """Per connection, its flowing pressure."""

    connection_oil_rates: NumberArray[OneDimension]
    """Per connection, its reservoir-condition oil rate."""

    connection_water_rates: NumberArray[OneDimension]
    """Per connection, its reservoir-condition water rate."""

    connection_gas_rates: NumberArray[OneDimension]
    """Per connection, its reservoir-condition gas rate."""

    well_index_scratch: NumberArray[OneDimension]
    """Per connection, room for the well index a solve changes as it iterates."""


@numba.njit(cache=True)
def slice_connection_fluids(
    fluids: ConnectionFluids, start: Integer, stop: Integer
) -> ConnectionFluids:
    """
    The connections `start` up to `stop` of stacked connection fluids, as views.

    :param fluids: Connection fluids of many wells, one after another.
    :param start: First connection.
    :param stop: One past the last connection.
    :returns: The fluids of those connections. Nothing is copied.
    """
    return ConnectionFluids(
        pressures=fluids.pressures[start:stop],
        gas_liquid_surface_tensions=fluids.gas_liquid_surface_tensions[start:stop],
        oil_densities=fluids.oil_densities[start:stop],
        water_densities=fluids.water_densities[start:stop],
        gas_densities=fluids.gas_densities[start:stop],
        oil_viscosities=fluids.oil_viscosities[start:stop],
        water_viscosities=fluids.water_viscosities[start:stop],
        gas_viscosities=fluids.gas_viscosities[start:stop],
        oil_saturations=fluids.oil_saturations[start:stop],
        water_saturations=fluids.water_saturations[start:stop],
        gas_saturations=fluids.gas_saturations[start:stop],
        oil_mobilities=fluids.oil_mobilities[start:stop],
        water_mobilities=fluids.water_mobilities[start:stop],
        gas_mobilities=fluids.gas_mobilities[start:stop],
        oil_formation_volume_factors=fluids.oil_formation_volume_factors[start:stop],
        water_formation_volume_factors=fluids.water_formation_volume_factors[start:stop],
        gas_formation_volume_factors=fluids.gas_formation_volume_factors[start:stop],
        solution_gas_oil_ratios=fluids.solution_gas_oil_ratios[start:stop],
        vaporized_oil_gas_ratios=fluids.vaporized_oil_gas_ratios[start:stop],
    )


@numba.njit(cache=True)
def solve_at(
    settings: CompiledSettings,
    wellbores: CompiledWellbores,
    row: Integer,
    workspace: PerforationWorkspace,
    reference_depth: Number,
    reference_pressure: Number,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
):
    """
    Solves a well's connection pressures and rates at a reference pressure, with the solver its
    correlation and the run's connection pressure mode call for.

    :param settings: Numbers shared by every well.
    :param wellbores: The correlation of every well.
    :param row: The well's row.
    :param workspace: The well's perforation workspace.
    :param reference_depth: The well's bottom-hole pressure datum.
    :param reference_pressure: The pressure at that datum.
    :param relevant_oil: Whether oil counts toward the solve's total rate.
    :param relevant_water: Whether water counts toward it.
    :param relevant_gas: Whether gas counts toward it.
    :param is_injector: Whether the well is an injector.
    :returns: The `FixedPointResult`.
    """
    tag = wellbores.tags[row]
    if tag >= 0:
        index = wellbores.indices[row]
        scale = wellbores.unit_scales[row]
        if tag == HOMOGENEOUS:
            return solve_homogeneous(
                model=wellbores.homogeneous[index],
                reference_depth=reference_depth,
                reference_pressure=reference_pressure,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        if tag == BEGGS_AND_BRILL:
            return solve_beggs_and_brill(
                model=wellbores.beggs_and_brill[index],
                reference_depth=reference_depth,
                reference_pressure=reference_pressure,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        if tag == HAGEDORN_BROWN:
            return solve_hagedorn_brown(
                model=wellbores.hagedorn_brown[index],
                reference_depth=reference_depth,
                reference_pressure=reference_pressure,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        if tag == GRAY:
            return solve_gray(
                model=wellbores.gray[index],
                reference_depth=reference_depth,
                reference_pressure=reference_pressure,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        return solve_woldesemayat_ghajar(
            model=wellbores.woldesemayat_ghajar[index],
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=scale,
            radial_flow_constant=settings.radial_flow_constant,
            max_iterations=settings.max_iterations,
            tolerance=settings.tolerance,
            tubing_fluid_table=None,
        )
    if settings.pressure_kind == PRESSURE_HYDROSTATIC:
        return solve_hydrostatic(
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            pressure_per_density_and_depth=settings.hydrostatic_scale,
            radial_flow_constant=settings.radial_flow_constant,
            max_iterations=settings.max_iterations,
            tolerance=settings.tolerance,
        )
    return solve_uniform(
        reference_pressure=reference_pressure,
        workspace=workspace,
        relevant_oil=relevant_oil,
        relevant_water=relevant_water,
        relevant_gas=relevant_gas,
        is_injector=is_injector,
        radial_flow_constant=settings.radial_flow_constant,
        max_iterations=settings.max_iterations,
        tolerance=settings.tolerance,
    )


@numba.njit(cache=True)
def bisect_at(
    settings: CompiledSettings,
    wellbores: CompiledWellbores,
    row: Integer,
    workspace: PerforationWorkspace,
    reference_depth: Number,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: bool,
):
    """
    Finds the reference pressure that gives a well a target rate, with the bisector its
    correlation and the run's connection pressure mode call for.

    :param settings: Numbers shared by every well.
    :param wellbores: The correlation of every well.
    :param row: The well's row.
    :param workspace: The well's perforation workspace.
    :param reference_depth: The well's bottom-hole pressure datum.
    :param relevant_oil: Whether oil counts toward the rate.
    :param relevant_water: Whether water counts toward it.
    :param relevant_gas: Whether gas counts toward it.
    :param is_injector: Whether the well is an injector.
    :param target: The target rate.
    :param low: Lowest reference pressure to search.
    :param high: Highest reference pressure to search.
    :param reservoir_condition: Whether the target is a reservoir-condition rate.
    :returns: The `BisectionResult`.
    """
    tag = wellbores.tags[row]
    if tag >= 0:
        index = wellbores.indices[row]
        scale = wellbores.unit_scales[row]
        if tag == HOMOGENEOUS:
            return bisect_homogeneous(
                model=wellbores.homogeneous[index],
                reference_depth=reference_depth,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                target=target,
                low=low,
                high=high,
                reservoir_condition=reservoir_condition,
                max_bisections=settings.max_bisections,
                rate_tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        if tag == BEGGS_AND_BRILL:
            return bisect_beggs_and_brill(
                model=wellbores.beggs_and_brill[index],
                reference_depth=reference_depth,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                target=target,
                low=low,
                high=high,
                reservoir_condition=reservoir_condition,
                max_bisections=settings.max_bisections,
                rate_tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        if tag == HAGEDORN_BROWN:
            return bisect_hagedorn_brown(
                model=wellbores.hagedorn_brown[index],
                reference_depth=reference_depth,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                target=target,
                low=low,
                high=high,
                reservoir_condition=reservoir_condition,
                max_bisections=settings.max_bisections,
                rate_tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        if tag == GRAY:
            return bisect_gray(
                model=wellbores.gray[index],
                reference_depth=reference_depth,
                workspace=workspace,
                relevant_oil=relevant_oil,
                relevant_water=relevant_water,
                relevant_gas=relevant_gas,
                is_injector=is_injector,
                hydrostatic_divisor=scale,
                radial_flow_constant=settings.radial_flow_constant,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                target=target,
                low=low,
                high=high,
                reservoir_condition=reservoir_condition,
                max_bisections=settings.max_bisections,
                rate_tolerance=settings.tolerance,
                tubing_fluid_table=None,
            )
        return bisect_woldesemayat_ghajar(
            model=wellbores.woldesemayat_ghajar[index],
            reference_depth=reference_depth,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=scale,
            radial_flow_constant=settings.radial_flow_constant,
            max_iterations=settings.max_iterations,
            tolerance=settings.tolerance,
            target=target,
            low=low,
            high=high,
            reservoir_condition=reservoir_condition,
            max_bisections=settings.max_bisections,
            rate_tolerance=settings.tolerance,
            tubing_fluid_table=None,
        )
    if settings.pressure_kind == PRESSURE_HYDROSTATIC:
        return bisect_hydrostatic(
            reference_depth=reference_depth,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            pressure_per_density_and_depth=settings.hydrostatic_scale,
            radial_flow_constant=settings.radial_flow_constant,
            max_iterations=settings.max_iterations,
            tolerance=settings.tolerance,
            target=target,
            low=low,
            high=high,
            reservoir_condition=reservoir_condition,
            max_bisections=settings.max_bisections,
            rate_tolerance=settings.tolerance,
        )
    return bisect_uniform(
        workspace=workspace,
        relevant_oil=relevant_oil,
        relevant_water=relevant_water,
        relevant_gas=relevant_gas,
        is_injector=is_injector,
        radial_flow_constant=settings.radial_flow_constant,
        max_iterations=settings.max_iterations,
        tolerance=settings.tolerance,
        target=target,
        low=low,
        high=high,
        reservoir_condition=reservoir_condition,
        max_bisections=settings.max_bisections,
        rate_tolerance=settings.tolerance,
    )


@numba.njit(cache=True)
def get_quantity_phases(quantity: Integer) -> tuple[bool, bool, bool]:
    """
    Which phases a rate quantity or a producer's rate control mode sums over.

    :param quantity: `RateQuantityTag` or `ProducerControlModeTag` rate mode, which share values.
    :returns: Whether oil, water and gas count.
    """
    if quantity == 0:
        return True, False, False
    if quantity == 1:
        return False, True, False
    if quantity == 2:
        return False, False, True
    if quantity == 3:
        return True, True, False
    return True, True, True


@numba.njit(cache=True)
def get_injected_phases(phase: Integer) -> tuple[bool, bool, bool]:
    """
    Which phase an injector's rate counts.

    :param phase: `FluidPhaseTag` of the injected phase.
    :returns: Whether oil, water and gas count.
    """
    return phase == 0, phase == 1, phase == 2


@numba.njit(cache=True)
def resolve_well(
    row: Integer,
    inputs: CompiledWellInputs,
    wellbores: CompiledWellbores,
    fluids: ConnectionFluids,
    outputs: CompiledWellOutputs,
    settings: CompiledSettings,
) -> Integer:
    """
    Resolves one well that is fully open and under BHP or rate control, as `resolve_well_control`
    does, and writes the result into `outputs`.

    :param row: The well's row.
    :param inputs: What is read about every well.
    :param wellbores: The correlation of every well.
    :param fluids: Stacked reservoir conditions at the connections of every well in the pass.
    :param outputs: Where the result is written.
    :param settings: Numbers shared by every well.
    :returns: `RESOLVED`, or `FAILED` if a solve did not converge.
    """
    start = inputs.perforation_offsets[row]
    stop = inputs.perforation_offsets[row + 1]
    fluid_start = inputs.fluid_starts[row]
    cell = slice_connection_fluids(fluids, fluid_start, fluid_start + (stop - start))
    static_well_indices = inputs.well_indices[start:stop]
    working_well_indices = outputs.well_index_scratch[start:stop]
    working_well_indices[:] = static_well_indices
    connection_gas_rates = outputs.connection_gas_rates[start:stop]
    connection_gas_rates[:] = 0.0
    workspace = PerforationWorkspace(
        well_indices=working_well_indices,
        static_well_indices=static_well_indices,
        connection_conductivities=inputs.connection_conductivities[start:stop],
        d_factor=inputs.d_factors[row],
        allows_crossflow=inputs.crossflow_flags[row] == 1,
        connection_fluids=cell,
        representative_depths=inputs.representative_depths[start:stop],
        representative_measured_depths=inputs.representative_measured_depths[start:stop],
        reference_measured_depth=inputs.reference_measured_depths[row],
        inclinations_from_vertical=inputs.inclinations_from_vertical[start:stop],
        connection_pressures=outputs.connection_pressures[start:stop],
        connection_oil_rates=outputs.connection_oil_rates[start:stop],
        connection_water_rates=outputs.connection_water_rates[start:stop],
        connection_gas_rates=connection_gas_rates,
        dissolved_gas_ratios=cell.solution_gas_oil_ratios,
        vaporized_oil_ratios=cell.vaporized_oil_gas_ratios,
        stb_to_volume_factor=settings.stb_to_volume_factor,
    )
    reference_depth = inputs.reference_depths[row]
    is_injector = inputs.well_kinds[row] == INJECTOR_KIND
    mode = inputs.control_modes[row]

    highest_cell_pressure = cell.pressures.max()
    if is_injector:
        min_pressure = cell.pressures.min()
        max_pressure = settings.injector_bracket_multiplier * highest_cell_pressure
        relevant_oil, relevant_water, relevant_gas = get_injected_phases(
            inputs.injected_phases[row]
        )
    else:
        min_pressure = settings.producer_bhp_floor
        max_pressure = highest_cell_pressure
        relevant_oil, relevant_water, relevant_gas = True, True, True

    # The phases the well is solved for once its BHP is known, which are the ones its limits are
    # applied under too.
    if is_injector and mode != INJECTOR_BHP:
        target_oil, target_water, target_gas = relevant_oil, relevant_water, relevant_gas
        reservoir_target = mode == INJECTOR_RESERVOIR_RATE
    elif is_injector:
        target_oil, target_water, target_gas = relevant_oil, relevant_water, relevant_gas
        reservoir_target = False
    else:
        target_oil, target_water, target_gas = get_quantity_phases(mode)
        reservoir_target = mode == PRODUCER_RESERVOIR_RATE

    if (is_injector and mode == INJECTOR_BHP) or (not is_injector and mode == PRODUCER_BHP):
        bhp = inputs.target_bhps[row]
    else:
        bisected = bisect_at(
            settings,
            wellbores,
            row,
            workspace,
            reference_depth,
            target_oil,
            target_water,
            target_gas,
            is_injector,
            inputs.target_rates[row],
            min_pressure,
            max_pressure,
            reservoir_target,
        )
        if not bisected.result.converged:
            return FAILED
        bhp = bisected.bhp
    result = solve_at(
        settings,
        wellbores,
        row,
        workspace,
        reference_depth,
        bhp,
        relevant_oil,
        relevant_water,
        relevant_gas,
        is_injector,
    )
    if not result.converged:
        return FAILED
    rates = PhaseValues(oil=result.oil_rate, water=result.water_rate, gas=result.gas_rate)
    surface_rates = PhaseValues(
        oil=result.surface_oil_rate,
        water=result.surface_water_rate,
        gas=result.surface_gas_rate,
    )

    # Limits, as `apply_limits` applies them.
    limits = CompiledLimits(
        well_offsets=inputs.limit_well_offsets,
        kinds=inputs.limit_kinds,
        quantities=inputs.limit_quantities,
        min_values=inputs.limit_min_values,
        max_values=inputs.limit_max_values,
        workover_actions=inputs.limit_workover_actions,
        end_run_flags=inputs.limit_end_run_flags,
    )
    limits_start = limits.well_offsets[row]
    limits_end = limits.well_offsets[row + 1]
    has_bound = False
    for limit_row in range(limits_start, limits_end):
        if limits.kinds[limit_row] != LIMIT_ECONOMIC:
            has_bound = True
    saved = (stop - start) if has_bound else 0
    nominal_oil = np.empty(saved, dtype=outputs.connection_oil_rates.dtype)
    nominal_water = np.empty(saved, dtype=outputs.connection_oil_rates.dtype)
    nominal_gas = np.empty(saved, dtype=outputs.connection_oil_rates.dtype)
    nominal_pressures = np.empty(saved, dtype=outputs.connection_oil_rates.dtype)
    if has_bound:
        nominal_oil[:] = workspace.connection_oil_rates
        nominal_water[:] = workspace.connection_water_rates
        nominal_gas[:] = workspace.connection_gas_rates
        nominal_pressures[:] = workspace.connection_pressures

    governing_bhp = bhp
    governing_row = UNSET_INT
    found = False
    for limit_row in range(limits_start, limits_end):
        kind = limits.kinds[limit_row]
        if kind == LIMIT_ECONOMIC:
            continue
        bound = bhp
        violated = False
        if kind == LIMIT_BHP:
            violated, bound = find_bhp_bound(
                limits.min_values[limit_row], limits.max_values[limit_row], bhp, is_injector
            )
        elif kind == LIMIT_RATE:
            quantity = limits.quantities[limit_row]
            limit_oil, limit_water, limit_gas = get_quantity_phases(quantity)
            reservoir_quantity = quantity == RESERVOIR_QUANTITY
            probe = solve_at(
                settings,
                wellbores,
                row,
                workspace,
                reference_depth,
                bhp,
                limit_oil,
                limit_water,
                limit_gas,
                is_injector,
            )
            if not probe.converged:
                return FAILED
            if reservoir_quantity:
                current_rate = probe.oil_rate + probe.water_rate + probe.gas_rate
            else:
                current_rate = (
                    probe.surface_oil_rate + probe.surface_water_rate + probe.surface_gas_rate
                )
            if current_rate > limits.max_values[limit_row]:
                bisected = bisect_at(
                    settings,
                    wellbores,
                    row,
                    workspace,
                    reference_depth,
                    limit_oil,
                    limit_water,
                    limit_gas,
                    is_injector,
                    limits.max_values[limit_row],
                    min_pressure,
                    max_pressure,
                    reservoir_quantity,
                )
                if not bisected.result.converged:
                    return FAILED
                bound, violated = bisected.bhp, True
        if violated:
            if not found:
                governing_bhp, governing_row, found = bound, limit_row, True
            elif is_injector and bound < governing_bhp:
                governing_bhp, governing_row = bound, limit_row
            elif (not is_injector) and bound > governing_bhp:
                governing_bhp, governing_row = bound, limit_row

    if not found:
        if has_bound:
            workspace.connection_oil_rates[:] = nominal_oil
            workspace.connection_water_rates[:] = nominal_water
            workspace.connection_gas_rates[:] = nominal_gas
            workspace.connection_pressures[:] = nominal_pressures
    else:
        bhp = governing_bhp
        result = solve_at(
            settings,
            wellbores,
            row,
            workspace,
            reference_depth,
            bhp,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
        )
        if not result.converged:
            return FAILED
        rates = PhaseValues(oil=result.oil_rate, water=result.water_rate, gas=result.gas_rate)
        surface_rates = PhaseValues(
            oil=result.surface_oil_rate,
            water=result.surface_water_rate,
            gas=result.surface_gas_rate,
        )

    economic_row = check_economic_violation(limits, limits_start, limits_end, surface_rates, True)
    shut_in = economic_row != UNSET_INT
    if shut_in:
        governing_row = economic_row

    reported_oil, reported_water, reported_gas = rates.oil, rates.water, rates.gas
    reported_surface_oil = surface_rates.oil
    reported_surface_water = surface_rates.water
    reported_surface_gas = surface_rates.gas
    if shut_in:
        reported_oil, reported_water, reported_gas = 0.0, 0.0, 0.0
        reported_surface_oil, reported_surface_water, reported_surface_gas = 0.0, 0.0, 0.0
    elif is_injector and workspace.allows_crossflow:
        if (
            (workspace.connection_oil_rates < 0.0).any()
            or (workspace.connection_water_rates < 0.0).any()
            or (workspace.connection_gas_rates < 0.0).any()
        ):
            summed, summed_surface = sum_connection_rates(workspace, True)
            reported_oil, reported_water, reported_gas = summed.oil, summed.water, summed.gas
            reported_surface_oil = summed_surface.oil
            reported_surface_water = summed_surface.water
            reported_surface_gas = summed_surface.gas

    outputs.bhps[row] = bhp
    outputs.oil_rates[row] = reported_oil
    outputs.water_rates[row] = reported_water
    outputs.gas_rates[row] = reported_gas
    outputs.surface_oil_rates[row] = reported_surface_oil
    outputs.surface_water_rates[row] = reported_surface_water
    outputs.surface_gas_rates[row] = reported_surface_gas
    outputs.active_limit_rows[row] = governing_row
    outputs.economic_shutins[row] = 1 if shut_in else 0
    if shut_in:
        workspace.connection_oil_rates[:] = 0.0
        workspace.connection_water_rates[:] = 0.0
        workspace.connection_gas_rates[:] = 0.0
    return RESOLVED


@numba.njit(cache=True, parallel=True, nogil=True)
def resolve_wells_kernel(
    rows: IntArray[OneDimension],
    resolutions: IntArray[OneDimension],
    inputs: CompiledWellInputs,
    wellbores: CompiledWellbores,
    fluids: ConnectionFluids,
    outputs: CompiledWellOutputs,
    settings: CompiledSettings,
) -> None:
    """
    Resolves the wells in `rows`, in parallel, each as `resolve_well` does.

    :param rows: The rows of the wells to resolve.
    :param resolutions: Per well, receives `RESOLVED` or `FAILED` for each well in `rows`.
    :param inputs: What is read about every well.
    :param wellbores: The correlation of every well.
    :param fluids: Stacked reservoir conditions at the connections of those wells.
    :param outputs: Where the results are written.
    :param settings: Numbers shared by every well.
    """
    for index in numba.prange(len(rows)):
        row = rows[index]
        resolutions[row] = resolve_well(row, inputs, wellbores, fluids, outputs, settings)


def build_compiled_wellbores(
    *, wellbore_models: typing.Sequence[WellBoreModel | None]
) -> CompiledWellbores:
    """
    Puts the hydraulics correlation of every well into arrays a compiled pass can index.

    :param wellbore_models: Per well, its `WellBoreModel`, or `None` for a well with none.
    :returns: The correlations as arrays.
    """
    n_wells = len(wellbore_models)
    tags = np.full(n_wells, -1, dtype=np.int64)
    indices = np.zeros(n_wells, dtype=np.int64)
    unit_scales = np.ones(n_wells, dtype=np.float64)
    options: list[list[tuple]] = [[] for _ in CORRELATION_CLASSES]
    for row, model in enumerate(wellbore_models):
        if model is None:
            continue
        tag = CORRELATION_TAGS[model.name]
        tags[row] = tag
        indices[row] = len(options[tag])
        options[tag].append(tuple(model.options))
        unit_scales[row] = get_hydrostatic_divisor(model.options.unit_system)

    records = []
    for tag, correlation in enumerate(CORRELATION_CLASSES):
        annotations = correlation.__annotations__
        dtype = np.dtype([
            (name, np.int64 if annotations[name] in (int, "int") else np.float64)
            for name in correlation._fields  # type: ignore[attr-defined]
        ])
        records.append(np.array(options[tag], dtype=dtype))
    return CompiledWellbores(
        tags=tags,
        indices=indices,
        unit_scales=unit_scales,
        homogeneous=records[HOMOGENEOUS],
        beggs_and_brill=records[BEGGS_AND_BRILL],
        hagedorn_brown=records[HAGEDORN_BROWN],
        gray=records[GRAY],
        woldesemayat_ghajar=records[WOLDESEMAYAT_GHAJAR],
    )


def stack_connection_fluids(
    *, fluids: typing.Sequence[ConnectionFluids], dtype: np.dtype
) -> ConnectionFluids:
    """
    Joins the connection fluids of several wells into one, one well after another.

    :param fluids: The connection fluids of each well.
    :param dtype: Data type of the result.
    :returns: Connection fluids of every connection of every well.
    """
    return ConnectionFluids(
        *(
            np.concatenate([field[index] for field in fluids]).astype(dtype, copy=False)
            for index in range(len(ConnectionFluids._fields))
        )
    )


def resolve_wells_in_parallel(
    *,
    compiled_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    spec: WellControlSpec,
    get_tubing_performance: typing.Callable[[Integer], WellBoreModel | VFPTables | None],
    get_connection_fluids: typing.Callable[[Integer], ConnectionFluids],
    get_surface_sample: typing.Callable[[Integer], SurfaceSample | None] | None = None,
) -> IntArray[OneDimension]:
    """
    Resolves every well it can take in one compiled pass over all of them, in parallel, and
    reports the rows of the rest.

    A well is taken if it is in effect, fully open with every connection flowing, under BHP or
    rate control, has no THP limit, and has no VFP table or tubing fluid table. Wells that are
    stopped, under THP control, or whose solve does not converge are left for
    `resolve_well_control`, which also raises the error of one that does not converge.

    :param compiled_system: The compiled well system.
    :param workspace: This run's `WellsWorkspace`, updated in place for the wells taken.
    :param spec: Solver tunables.
    :param get_tubing_performance: Given a well row, its `WellBoreModel` or `VFPTables`, if any.
    :param get_connection_fluids: Given a well row, its flowing connections' conditions.
    :param get_surface_sample: Given a well row, its surface fluid properties, if any.
    :returns: The rows still to be resolved one at a time. Wells not in effect are not among them.
    :raises StopSimulation: If a well it resolved breached an economic limit flagged to end the
        run.
    """
    n_wells = len(compiled_system.names)
    dtype = np.dtype(get_dtype())
    in_effect = compiled_system.schedule_statuses != 0
    if not spec.use_compiled_resolution or workspace.bhps.dtype != dtype:
        return np.flatnonzero(in_effect)

    controls = compiled_system.controls
    perforations = compiled_system.perforations
    limits = controls.limits
    offsets = perforations.well_offsets
    is_injector = controls.well_kinds == INJECTOR_KIND

    control_modes = controls.control_modes.copy()
    target_rates = controls.target_rates.copy()
    group_modes = workspace.group_control_modes
    group_substituted = (group_modes != UNSET_INT) & (
        control_modes
        == np.where(
            is_injector, int(InjectorControlModeTag.GROUP), int(ProducerControlModeTag.GROUP)
        )
    )
    control_modes[group_substituted] = group_modes[group_substituted]
    target_rates[group_substituted] = workspace.group_target_rates[group_substituted]

    controllable = np.where(
        is_injector,
        np.isin(control_modes, (INJECTOR_RATE, INJECTOR_RESERVOIR_RATE, INJECTOR_BHP)),
        np.isin(control_modes, (*PRODUCER_RATE_MODES, PRODUCER_BHP)),
    )
    flowing = (perforations.completion_statuses == 1) & (perforations.schedule_statuses == 1)
    flowing_total = np.concatenate(([0], np.cumsum(flowing)))
    flowing_counts = flowing_total[offsets[1:]] - flowing_total[offsets[:-1]]
    all_flowing = (flowing_counts == np.diff(offsets)) & (flowing_counts > 0)
    thp_total = np.concatenate(([0], np.cumsum(limits.kinds == int(LimitKind.THP))))
    has_thp_limit = thp_total[limits.well_offsets[1:]] != thp_total[limits.well_offsets[:-1]]
    candidates = np.flatnonzero(
        in_effect
        & (compiled_system.open_statuses == 1)
        & controllable
        & all_flowing
        & ~has_thp_limit
    )

    counts = np.diff(offsets)
    taken: list[int] = []
    wellbore_models: list[WellBoreModel | None] = [None] * n_wells
    fluids_by_well: list[ConnectionFluids] = []
    reports_thp: list[tuple[int, WellTubing, SurfaceSample]] = []
    for row in candidates:
        tubing = resolve_well_tubing(
            tubing_performance=get_tubing_performance(row),
            compiled_system=compiled_system,
            well_row=row,
        )
        wellbore = tubing.wellbore
        if tubing.vfp_table is not None or tubing.uses_tubing_fluid_table:
            continue
        if wellbore is None and spec.connection_pressure_mode == ConnectionPressureMode.HYDRAULIC:
            continue
        if wellbore is not None and wellbore.name not in CORRELATION_TAGS:
            continue
        taken.append(row)
        wellbore_models[row] = wellbore
        fluids = get_connection_fluids(row)
        if len(fluids.pressures) != counts[row]:
            raise ValidationError(
                f"`resolve_wells_in_parallel`: `well_row` {row} has {counts[row]} flowing "
                f"connections but was given {len(fluids.pressures)} connections in "
                "`connection_fluids`. These must match 1:1."
            )
        fluids_by_well.append(fluids)
        surface_sample = get_surface_sample(row) if get_surface_sample is not None else None
        if wellbore is not None and surface_sample is not None:
            reports_thp.append((row, tubing, surface_sample))
    not_taken = np.setdiff1d(np.flatnonzero(in_effect), taken, assume_unique=True)
    if not taken:
        return not_taken

    fluid_starts = np.full(n_wells, -1, dtype=np.int64)
    fluid_starts[taken] = np.concatenate(([0], np.cumsum(counts[taken])[:-1]))
    inputs = CompiledWellInputs(
        perforation_offsets=offsets,
        well_indices=np.asarray(perforations.well_indices, dtype=dtype),
        representative_depths=np.asarray(perforations.representative_depths, dtype=dtype),
        representative_measured_depths=np.asarray(
            perforations.representative_measured_depths, dtype=dtype
        ),
        inclinations_from_vertical=np.asarray(
            perforations.inclinations_from_vertical, dtype=dtype
        ),
        connection_conductivities=np.asarray(perforations.connection_conductivities, dtype=dtype),
        reference_depths=compiled_system.reference_depths,
        reference_measured_depths=compiled_system.reference_measured_depths,
        d_factors=compiled_system.d_factors,
        crossflow_flags=compiled_system.crossflow_flags,
        well_kinds=controls.well_kinds,
        injected_phases=controls.injected_phases,
        control_modes=control_modes,
        target_rates=target_rates,
        target_bhps=controls.target_bhps,
        limit_well_offsets=limits.well_offsets,
        limit_kinds=limits.kinds,
        limit_quantities=limits.quantities,
        limit_min_values=limits.min_values,
        limit_max_values=limits.max_values,
        limit_workover_actions=limits.workover_actions,
        limit_end_run_flags=limits.end_run_flags,
        fluid_starts=fluid_starts,
    )
    outputs = CompiledWellOutputs(
        bhps=workspace.bhps,
        oil_rates=workspace.oil_rates,
        water_rates=workspace.water_rates,
        gas_rates=workspace.gas_rates,
        surface_oil_rates=workspace.surface_oil_rates,
        surface_water_rates=workspace.surface_water_rates,
        surface_gas_rates=workspace.surface_gas_rates,
        active_limit_rows=workspace.active_limit_rows,
        economic_shutins=workspace.economic_shutins,
        connection_pressures=workspace.connection_pressures,
        connection_oil_rates=workspace.connection_oil_rates,
        connection_water_rates=workspace.connection_water_rates,
        connection_gas_rates=workspace.connection_gas_rates,
        well_index_scratch=workspace.well_index_scratch,
    )
    resolutions = np.zeros(n_wells, dtype=np.int64)
    settings = CompiledSettings(
        pressure_kind=PRESSURE_HYDROSTATIC
        if spec.connection_pressure_mode == ConnectionPressureMode.HYDROSTATIC
        else PRESSURE_UNIFORM,
        radial_flow_constant=spec.radial_flow_constant,
        hydrostatic_scale=get_pressure_per_density_and_depth(spec.unit_system),
        max_iterations=spec.max_fixed_point_iterations,
        tolerance=spec.rate_convergence_tolerance,
        max_bisections=spec.max_bisection_iterations,
        producer_bhp_floor=spec.producer_bhp_floor,
        injector_bracket_multiplier=dtype.type(spec.injector_bhp_bracket_multiplier),
        stb_to_volume_factor=get_stb_to_volume_factor(compiled_system.unit_system),
    )
    resolve_wells_kernel(
        np.asarray(taken, dtype=np.int64),
        resolutions,
        inputs,
        build_compiled_wellbores(wellbore_models=wellbore_models),
        stack_connection_fluids(fluids=fluids_by_well, dtype=dtype),
        outputs,
        settings,
    )

    unresolved = [row for row in taken if resolutions[row] != RESOLVED]
    for row in taken:
        if resolutions[row] == RESOLVED and outputs.economic_shutins[row]:
            if limits.end_run_flags[outputs.active_limit_rows[row]]:
                raise StopSimulation(
                    f"Well {compiled_system.names[row]!r} breached an economic limit "
                    "flagged to end the run."
                )
    for row, tubing, surface_sample in reports_thp:
        if resolutions[row] != RESOLVED:
            continue
        start, stop = offsets[row], offsets[row + 1]
        oil = workspace.connection_oil_rates[start:stop]
        water = workspace.connection_water_rates[start:stop]
        gas = workspace.connection_gas_rates[start:stop]
        if (
            is_injector[row]
            and compiled_system.crossflow_flags[row] == 1
            and not workspace.economic_shutins[row]
            and ((oil < 0.0).any() or (water < 0.0).any() or (gas < 0.0).any())
        ):
            unresolved.append(row)
            continue
        report_tubing_head_pressure(
            tubing=tubing,
            reference_depth=compiled_system.reference_depths[row],
            bhp=workspace.bhps[row],
            rates=PhaseValues(
                oil=workspace.oil_rates[row],
                water=workspace.water_rates[row],
                gas=workspace.gas_rates[row],
            ),
            surface_rates=PhaseValues(
                oil=workspace.surface_oil_rates[row],
                water=workspace.surface_water_rates[row],
                gas=workspace.surface_gas_rates[row],
            ),
            surface_sample=surface_sample,
            is_injector=bool(is_injector[row]),
            connection_rates=ConnectionRates(oil=oil, water=water, gas=gas),
            well_row=row,
            workspace=workspace,
        )
    return np.union1d(not_taken, unresolved)
