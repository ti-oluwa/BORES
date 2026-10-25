"""Mode-specific well-control solvers: target -> nominal (bhp, phase_rates)."""

import typing

from bores.errors import ValidationError, WellConvergenceError
from bores.types import Boolean, Integer, Number, NumberArray, OneDimension
from bores.wells.compile import (
    FluidPhaseTag,
    InjectorControlModeTag,
    ProducerControlModeTag,
    RateQuantityTag,
)
from bores.wells.hydraulics.base import (
    SurfaceSample,
    get_hydrostatic_divisor,
)
from bores.wells.hydraulics.hydrostatic import get_pressure_per_density_and_depth
from bores.wells.resolution.fixed_point import (
    FIXED_POINT_SOLVERS,
    RATE_BISECTORS,
    FixedPointBisector,
    FixedPointResult,
    FixedPointSolver,
    HydraulicBisector,
    HydraulicSolver,
    HydrostaticBisector,
    HydrostaticSolver,
    UniformBisector,
    UniformSolver,
    bisect_hydrostatic,
    bisect_uniform,
    solve_hydrostatic,
    solve_uniform,
)
from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
from bores.wells.resolution.tubing import WellTubing, compute_tubing_head_pressure
from bores.wells.state import PhaseValues
from bores.wells.workspace import (
    PerforationWorkspace,
    get_connection_rates,
)

__all__ = [
    "PRODUCER_RATE_MODE_PHASES",
    "RATE_QUANTITY_PHASES",
    "get_pressure_bracket",
    "phase_mask",
    "solve_bhp",
    "solve_bhp_injector",
    "solve_bhp_producer",
    "solve_connection_pressures_and_rates",
    "solve_rate_injector",
    "solve_rate_producer",
]


PRODUCER_RATE_MODE_PHASES: dict[Integer, PhaseValues] = {
    ProducerControlModeTag.OIL_RATE: PhaseValues(oil=1.0, water=0.0, gas=0.0),
    ProducerControlModeTag.WATER_RATE: PhaseValues(oil=0.0, water=1.0, gas=0.0),
    ProducerControlModeTag.GAS_RATE: PhaseValues(oil=0.0, water=0.0, gas=1.0),
    ProducerControlModeTag.LIQUID_RATE: PhaseValues(oil=1.0, water=1.0, gas=0.0),
    ProducerControlModeTag.RESERVOIR_VOLUME_RATE: PhaseValues(oil=1.0, water=1.0, gas=1.0),
}
"""Which phases count toward a producer's rate target, by `ProducerControlModeTag`."""

RATE_QUANTITY_PHASES: dict[Integer, PhaseValues] = {
    RateQuantityTag.OIL: PhaseValues(oil=1.0, water=0.0, gas=0.0),
    RateQuantityTag.WATER: PhaseValues(oil=0.0, water=1.0, gas=0.0),
    RateQuantityTag.GAS: PhaseValues(oil=0.0, water=0.0, gas=1.0),
    RateQuantityTag.LIQUID: PhaseValues(oil=1.0, water=1.0, gas=0.0),
    RateQuantityTag.RESERVOIR: PhaseValues(oil=1.0, water=1.0, gas=1.0),
}
"""Which phases a `RateLimit` sums over, by `RateQuantityTag`."""

ALL_PHASES = PhaseValues(oil=1.0, water=1.0, gas=1.0)
"""Every phase relevant - a producer's rate report includes oil, water, and gas."""


def phase_mask(phase_tag: Integer) -> PhaseValues:
    """
    Builds a single-phase relevant-phases mask from a `FluidPhaseTag`.

    :param phase_tag: `FluidPhaseTag` value, e.g. an injector's `injected_phase`.
    :returns: `PhaseValues` with `1.0` for `phase_tag`'s phase, `0.0` for the other two.
    """
    return PhaseValues(
        oil=1.0 if phase_tag == FluidPhaseTag.OIL else 0.0,
        water=1.0 if phase_tag == FluidPhaseTag.WATER else 0.0,
        gas=1.0 if phase_tag == FluidPhaseTag.GAS else 0.0,
    )


def get_pressure_bracket(
    workspace: PerforationWorkspace,
    *,
    is_injector: Boolean,
    control_spec: WellControlSpec,
) -> tuple[Number, Number]:
    """
    Builds a default `(min_pressure, max_pressure)` BHP bisection bracket.

    A producer's BHP can't usefully exceed the highest connected-cell
    pressure as there's no drawdown left to give a rate above that. An
    injector's upper bound is reservoir pressure scaled by
    `control_spec.injector_bhp_bracket_multiplier`, giving room above
    reservoir pressure to actually inject.

    :param workspace: This well's workspace, including reservoir pressure at each active connection.
    :param is_injector: Selects which bracket shape to use.
    :param control_spec: Supplies `producer_bhp_floor`/`injector_bhp_bracket_multiplier`.
    :returns: `(min_pressure, max_pressure)`.
    """
    cell_pressures = workspace.connection_fluids.pressures
    if is_injector:
        return min(cell_pressures), control_spec.injector_bhp_bracket_multiplier * max(
            cell_pressures
        )
    return control_spec.producer_bhp_floor, max(cell_pressures)


class FixedPointSetup(typing.NamedTuple):
    """
    What a well's fixed-point solve and rate bisection need, by how its connection pressures
    are found.
    """

    kind: typing.Literal["hydraulic", "hydrostatic", "uniform"]
    """
    How the connection pressures are found: from a hydraulics correlation, from the weight of
    the fluid between connections, or as the reference pressure unchanged.
    """

    model: typing.Any
    """The correlation's options, for the `"hydraulic"` kind. `None` otherwise."""

    unit_scale: Number | None
    """
    The hydrostatic divisor for `"hydraulic"`, or the pressure of a unit column for
    `"hydrostatic"`. `None` for `"uniform"`.
    """

    solve: FixedPointSolver
    """The compiled fixed-point solver."""

    bisect: FixedPointBisector
    """The compiled rate bisection, with the same pressure profile as `solve`."""


def get_fixed_point_setup(*, tubing: WellTubing, control_spec: WellControlSpec) -> FixedPointSetup:
    """
    Chooses how a well's connection pressures are found, and the compiled solver for it.

    :param tubing: This well's tubing performance.
    :param control_spec: Supplies `connection_pressure_mode` and the unit system.
    :returns: The kind, and what the compiled solvers need for it.
    :raises ValidationError: If the well has no `WellBoreModel` and `connection_pressure_mode` is
        `HYDRAULIC`, or its model is unknown.
    """
    wellbore = tubing.wellbore
    pressure_mode = control_spec.connection_pressure_mode
    if wellbore is not None:
        solve = FIXED_POINT_SOLVERS.get(wellbore.name)
        bisect = RATE_BISECTORS.get(wellbore.name)
        if solve is None or bisect is None:
            raise ValidationError(
                f"Unknown wellbore model {wellbore.name!r}; expected one of "
                f"{sorted(FIXED_POINT_SOLVERS)}."
            )
        return FixedPointSetup(
            kind="hydraulic",
            model=wellbore.options,
            unit_scale=get_hydrostatic_divisor(wellbore.options.unit_system),
            solve=solve,
            bisect=bisect,
        )
    if pressure_mode == ConnectionPressureMode.HYDRAULIC:
        raise ValidationError(
            "This well has no `WellBoreModel` assigned, and "
            "`control_spec.connection_pressure_mode` is `HYDRAULIC`. "
            "A VFP table relates BHP to THP only and can't distribute pressure across "
            "connections. Assign a `WellBoreModel` to this well, or set "
            "`connection_pressure_mode` to `ConnectionPressureMode.HYDROSTATIC` to distribute "
            "pressure by the weight of the fluid between connections, or to "
            "`ConnectionPressureMode.UNIFORM_BHP` to apply the reference pressure unchanged "
            "at every connection."
        )
    if pressure_mode == ConnectionPressureMode.HYDROSTATIC:
        return FixedPointSetup(
            kind="hydrostatic",
            model=None,
            unit_scale=get_pressure_per_density_and_depth(control_spec.unit_system),
            solve=solve_hydrostatic,
            bisect=bisect_hydrostatic,
        )
    return FixedPointSetup(
        kind="uniform",
        model=None,
        unit_scale=None,
        solve=solve_uniform,
        bisect=bisect_uniform,
    )


def raise_if_not_converged(
    *,
    result: FixedPointResult,
    reference_pressure: Number,
    max_iterations: Integer,
    tolerance: Number,
) -> None:
    """
    Raises if a fixed-point solve didn't settle.

    :param result: What the solve ended with.
    :param reference_pressure: The reference pressure it was solved at.
    :param max_iterations: The iteration limit it ran under.
    :param tolerance: The relative tolerance it ran under.
    :raises WellConvergenceError: If `result` didn't converge.
    """
    if not result.converged:
        raise WellConvergenceError(
            "The connection pressures and rates did not converge in "
            f"{max_iterations} iterations at a reference pressure of "
            f"{reference_pressure}: on the last iteration the total rate went from "
            f"{result.previous_total_rate} to {result.total_rate}, against a relative tolerance of "
            f"{tolerance}. A rate high enough that friction or "
            "head takes the connection pressures past reservoir pressure can cause this."
        )


def solve_connection_pressures_and_rates(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    reference_pressure: Number,
    relevant_phases: PhaseValues,
    is_injector: Boolean,
    control_spec: WellControlSpec,
) -> tuple[NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Runs fixed-point iterations of connection flowing pressures against IPR-derived
    rates at a fixed `reference_pressure` (BHP), until the well-total
    relevant-phase rate stabilizes.

    Starts from static (zero-rate) connection pressures, derives IPR rates
    from those, feeds the resulting `phase_rates` back through the
    hydraulics correlation for updated flowing pressures, and repeats
    until the relative change in total rate is within
    `control_spec.rate_convergence_tolerance`, or
    `control_spec.max_fixed_point_iterations` is reached.
    `workspace.connection_pressures` is reused as the output buffer on
    every call rather than reallocated.

    Each iteration also corrects `workspace.well_indices` for non-Darcy
    flow from `workspace.static_well_indices` and the previous
    iteration's gas rates, before computing this iteration's rates. A
    well with no `d_factor` set is unaffected. This still runs when
    the well has no `WellBoreModel`, since non-Darcy correction doesn't depend on
    wellbore hydraulics.

    :param tubing: Hydraulics correlation for this well. If it has none,
        behavior depends on `control_spec.connection_pressure_mode`: with
        `UNIFORM_BHP`, `reference_pressure` is applied unchanged at every
        connection; with `HYDROSTATIC`, pressure follows the weight of the fluid
        between connections; with `HYDRAULIC` (the default), raises.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param reference_pressure: BHP held fixed while iterating.
    :param relevant_phases: Mask of the phases whose totals are returned, which are the ones
        counting toward the controlled quantity. A producer's per-connection rates, and so its
        pressure profile, always include every phase, since all of them flow. An injector
        flows only the phases in the mask.
    :param is_injector: Selects the drawdown sign convention.
    :param control_spec: Supplies the iteration cap, convergence tolerance,
        and `connection_pressure_mode`.
    :returns: `(connection_pressures, phase_rates, surface_phase_rates)`.
    :raises ValidationError: If the well has no `WellBoreModel` and
        `control_spec.connection_pressure_mode` is `HYDRAULIC`, or its model is unknown.
    :raises WellConvergenceError: If the connection pressures and rates don't settle within the
        iteration limit.
    """
    setup = get_fixed_point_setup(tubing=tubing, control_spec=control_spec)
    relevant_oil = relevant_phases.oil > 0.0
    relevant_water = relevant_phases.water > 0.0
    relevant_gas = relevant_phases.gas > 0.0
    radial_flow_constant = control_spec.radial_flow_constant
    max_iterations = control_spec.max_fixed_point_iterations
    tolerance = control_spec.rate_convergence_tolerance

    if setup.kind == "hydraulic":
        assert setup.unit_scale is not None
        solver = typing.cast(HydraulicSolver, setup.solve)
        result = solver(
            model=setup.model,
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=setup.unit_scale,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    elif setup.kind == "hydrostatic":
        assert setup.unit_scale is not None
        solver = typing.cast(HydrostaticSolver, setup.solve)
        result = solver(
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            pressure_per_density_and_depth=setup.unit_scale,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    else:
        solver = typing.cast(UniformSolver, setup.solve)
        result = solver(
            reference_pressure=reference_pressure,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    raise_if_not_converged(
        result=result,
        reference_pressure=reference_pressure,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return (
        workspace.connection_pressures,
        PhaseValues(oil=result.oil_rate, water=result.water_rate, gas=result.gas_rate),
        PhaseValues(
            oil=result.surface_oil_rate,
            water=result.surface_water_rate,
            gas=result.surface_gas_rate,
        ),
    )


def bisect_rate_bhp(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_phases: PhaseValues,
    is_injector: Boolean,
    target: Number,
    min_pressure: Number,
    max_pressure: Number,
    control_spec: WellControlSpec,
    target_rate_condition: typing.Literal["surface", "reservoir"],
) -> tuple[Number, NumberArray[OneDimension], PhaseValues]:
    """
    Bisects `reference_pressure` in `[min_pressure, max_pressure]` until the total rate of
    `relevant_phases` matches `target`, in one compiled call.

    See `solve_bhp` for what the bisection does and returns.

    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param relevant_phases: Mask the rate is summed over.
    :param is_injector: Selects the drawdown sign convention and the direction the rate moves in
        as BHP changes.
    :param target: The total rate to match.
    :param min_pressure: Lower bracket bound.
    :param max_pressure: Upper bracket bound.
    :param control_spec: Supplies the iteration caps and convergence tolerances.
    :param target_rate_condition: `"surface"` bisects against the surface-condition total,
        `"reservoir"` against the reservoir-condition one.
    :returns: `(bhp, connection_pressures, phase_rates)`, `phase_rates` reservoir-condition.
    :raises WellConvergenceError: If a solve inside the bisection doesn't converge.
    """
    setup = get_fixed_point_setup(tubing=tubing, control_spec=control_spec)
    relevant_oil = relevant_phases.oil > 0.0
    relevant_water = relevant_phases.water > 0.0
    relevant_gas = relevant_phases.gas > 0.0
    radial_flow_constant = control_spec.radial_flow_constant
    max_iterations = control_spec.max_fixed_point_iterations
    tolerance = control_spec.rate_convergence_tolerance
    assert control_spec.max_bisection_iterations > 0
    bisection = (
        target,
        min_pressure,
        max_pressure,
        target_rate_condition == "reservoir",
        control_spec.max_bisection_iterations,
        control_spec.rate_convergence_tolerance,
    )

    if setup.kind == "hydraulic":
        assert setup.unit_scale is not None
        bisector = typing.cast(HydraulicBisector, setup.bisect)
        bisected = bisector(
            setup.model,
            reference_depth,
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            setup.unit_scale,
            radial_flow_constant,
            max_iterations,
            tolerance,
            *bisection,
        )
    elif setup.kind == "hydrostatic":
        assert setup.unit_scale is not None
        bisector = typing.cast(HydrostaticBisector, setup.bisect)
        bisected = bisector(
            reference_depth,
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            setup.unit_scale,
            radial_flow_constant,
            max_iterations,
            tolerance,
            *bisection,
        )
    else:
        bisector = typing.cast(UniformBisector, setup.bisect)
        bisected = bisector(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
            max_iterations,
            tolerance,
            *bisection,
        )

    raise_if_not_converged(
        result=bisected.result,
        reference_pressure=bisected.bhp,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    result = bisected.result
    return (
        bisected.bhp,
        workspace.connection_pressures,
        PhaseValues(oil=result.oil_rate, water=result.water_rate, gas=result.gas_rate),
    )


def solve_bhp(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_phases: PhaseValues,
    is_injector: Boolean,
    target: Number,
    min_pressure: Number,
    max_pressure: Number,
    control_spec: WellControlSpec,
    metric: typing.Literal["rate", "thp"] = "rate",
    target_rate_condition: typing.Literal["surface", "reservoir"] = "surface",
    surface_sample: SurfaceSample | None = None,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues]:
    """
    Bisects `reference_pressure` in `[min_pressure, max_pressure]` until
    the resulting metric (`relevant_phases` total rate, or THP) matches `target`.

    Both metrics are monotonic in BHP, in a direction that depends on the
    metric and (for rate) the well type. For `metric="rate"`, lower BHP means
    a higher producer rate, higher BHP means a higher injector rate;
    For `metric="thp"`, higher BHP means higher THP, for both well types.

    `min_pressure`/`max_pressure` must bracket the target. Callers should pick
    these via `get_pressure_bracket` (or their own). Best-effort
    (no exception) if `target` isn't achievable within the bracket.
    Returns the closest bound reached after `control_spec.max_bisection_iterations`.

    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param relevant_phases: Mask the `"rate"` metric sums over.
    :param is_injector: Selects the drawdown sign convention and the
        direction `"rate"` moves in as BHP changes.
    :param target: Value the chosen metric is bisected toward.
    :param min_pressure: Lower bracket bound.
    :param max_pressure: Upper bracket bound.
    :param control_spec: Supplies the iteration cap and convergence tolerance.
    :param metric: `"rate"` (default) matches `relevant_phases`' total
        rate against `target`; `"thp"` matches tubing head pressure
        against `target` instead (requires `surface_sample`).
    :param target_rate_condition: `"surface"` (default) bisects against
        the surface-condition total; `"reservoir"` bisects against the
        raw reservoir-condition total instead. Ignored when `metric="thp"`.
    :param surface_sample: Required if `metric="thp"`.
    :returns: `(bhp, connection_pressures, phase_rates)`, `phase_rates`
        always reservoir-condition regardless of `target_rate_condition`.
    :raises ValidationError: If `metric="thp"` and `surface_sample` is `None`.
    """
    if metric == "thp" and surface_sample is None and tubing.vfp_table is None:
        raise ValidationError("solve_bhp(..., metric='thp') requires `surface_sample`.")

    if metric == "rate":
        return bisect_rate_bhp(
            tubing=tubing,
            reference_depth=reference_depth,
            workspace=workspace,
            relevant_phases=relevant_phases,
            is_injector=is_injector,
            target=target,
            min_pressure=min_pressure,
            max_pressure=max_pressure,
            control_spec=control_spec,
            target_rate_condition=target_rate_condition,
        )

    low, high = min_pressure, max_pressure
    average_pressure = 0.5 * (low + high)
    pressures = None
    phase_rates = PhaseValues(oil=0.0, water=0.0, gas=0.0)

    for _ in range(control_spec.max_bisection_iterations):
        average_pressure = 0.5 * (low + high)
        pressures, phase_rates, surface_phase_rates = solve_connection_pressures_and_rates(
            tubing=tubing,
            reference_depth=reference_depth,
            workspace=workspace,
            reference_pressure=average_pressure,
            relevant_phases=relevant_phases,
            is_injector=is_injector,
            control_spec=control_spec,
        )

        value = compute_tubing_head_pressure(
            tubing=tubing,
            reference_depth=reference_depth,
            reference_pressure=average_pressure,
            phase_rates=phase_rates,
            surface_phase_rates=surface_phase_rates,
            surface_sample=surface_sample,
            is_injector=is_injector,
            connection_rates=get_connection_rates(workspace=workspace),
        )

        if abs(value - target) <= control_spec.rate_convergence_tolerance * max(abs(target), 1.0):
            break

        if value < target:
            low = average_pressure
        else:
            high = average_pressure

    assert pressures is not None
    return average_pressure, pressures, phase_rates


def solve_rate_producer(
    *,
    control_mode: Integer,
    target_rate: Number,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    control_spec: WellControlSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Resolves a producer under `control_mode in {ORAT, WRAT, GRAT, LRAT, RESV}`.

    Nominal resolution only. Does not check limits; the caller applies
    those afterward via `wells.resolution.limits.apply_limits`.

    :param control_mode: `ProducerControlModeTag` value.
    :param target_rate: The rate target.
    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param control_spec: FixedPointSolver tunables.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)`,
        reservoir- and surface-condition rates for every phase.
    """
    relevant_phases = PRODUCER_RATE_MODE_PHASES[control_mode]
    min_pressure, max_pressure = get_pressure_bracket(
        workspace=workspace, is_injector=False, control_spec=control_spec
    )
    bhp, _, _ = solve_bhp(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        relevant_phases=relevant_phases,
        is_injector=False,
        target=target_rate,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        control_spec=control_spec,
        target_rate_condition="reservoir"
        if control_mode == ProducerControlModeTag.RESERVOIR_VOLUME_RATE
        else "surface",
    )
    connection_pressures, phase_rates, surface_phase_rates = solve_connection_pressures_and_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        reference_pressure=bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        control_spec=control_spec,
    )
    return bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_bhp_producer(
    *,
    target_bhp: Number,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    control_spec: WellControlSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Resolves a producer held at `target_bhp`; connection pressures and rates are the output.

    Computes the connection pressure profile and IPR-derived rates at the supplied BHP. A producer
    can flow oil, water, and gas simultaneously, so rates for all phases are returned.

    Nominal resolution only. Does not check limits; the caller applies those afterward via
    `wells.resolution.limits.apply_limits`.

    :param target_bhp: BHP held fixed while computing connection pressures and rates.
    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param control_spec: FixedPointSolver tunables, including fixed-point convergence settings and
        connection-pressure mode.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)`, with `bhp` equal to
        `target_bhp` and rates for oil, water, and gas at reservoir and surface conditions.
    """
    connection_pressures, phase_rates, surface_phase_rates = solve_connection_pressures_and_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        reference_pressure=target_bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        control_spec=control_spec,
    )
    return target_bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_rate_injector(
    *,
    control_mode: Integer,
    target_rate: Number,
    injected_phase: Integer,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    control_spec: WellControlSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Resolves an injector rate target by bisecting BHP. `control_mode` selects the rate quantity
    being matched, and `injected_phase` selects the only phase whose rate is nonzero.

    Nominal resolution only. Does not check limits; the caller applies those afterward via
    `wells.resolution.limits.apply_limits`.

    :param control_mode: `InjectorControlModeTag` value.
    :param target_rate: The rate target.
    :param injected_phase: `FluidPhaseTag` value of the injected phase.
    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param control_spec: FixedPointSolver tunables.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)` -
        reservoir- and surface-condition rates, `injected_phase` only nonzero.
    """
    relevant_phases = phase_mask(injected_phase)
    min_pressure, max_pressure = get_pressure_bracket(
        workspace=workspace, is_injector=True, control_spec=control_spec
    )
    bhp, _, _ = solve_bhp(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        relevant_phases=relevant_phases,
        is_injector=True,
        target=target_rate,
        min_pressure=min_pressure,
        max_pressure=max_pressure,
        control_spec=control_spec,
        target_rate_condition="reservoir"
        if control_mode == InjectorControlModeTag.RESERVOIR_VOLUME_RATE
        else "surface",
    )
    connection_pressures, phase_rates, surface_phase_rates = solve_connection_pressures_and_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        reference_pressure=bhp,
        relevant_phases=relevant_phases,
        is_injector=True,
        control_spec=control_spec,
    )
    return bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_bhp_injector(
    *,
    target_bhp: Number,
    injected_phase: Integer,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    control_spec: WellControlSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Resolves an injector held at `target_bhp`; connection pressures and rates are the output.

    Computes the connection pressure profile and allocates rates to `injected_phase` at the
    supplied BHP. The other phases have zero rates.

    Nominal resolution only. Does not check limits; the caller applies those afterward via
    `wells.resolution.limits.apply_limits`.

    :param target_bhp: BHP held fixed while computing connection pressures and rates.
    :param injected_phase: `FluidPhaseTag` value selecting the phase being injected.
    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param control_spec: FixedPointSolver tunables, including fixed-point convergence settings and
        connection-pressure mode.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)`, with `bhp` equal to
        `target_bhp` and only the injected phase nonzero in the reservoir- and surface-condition
        rates.
    """
    relevant_phases = phase_mask(injected_phase)
    connection_pressures, phase_rates, surface_phase_rates = solve_connection_pressures_and_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        reference_pressure=target_bhp,
        relevant_phases=relevant_phases,
        is_injector=True,
        control_spec=control_spec,
    )
    return target_bhp, connection_pressures, phase_rates, surface_phase_rates
