"""Mode-specific well-control solvers: target -> nominal (bhp, phase_rates)."""

import typing

import numpy as np

from bores.errors import ValidationError, WellConvergenceError
from bores.types import Integer, Number, NumberArray, OneDimension
from bores.wells.compile import (
    FluidPhaseTag,
    InjectorControlModeTag,
    ProducerControlModeTag,
    RateQuantityTag,
)
from bores.wells.hydraulics.base import (
    SurfaceFluidProperties,
    WellBoreModel,
    compute_perforation_pressures_from_fluids,
)
from bores.wells.resolution.hydrostatic import compute_hydrostatic_connection_pressures
from bores.wells.resolution.spec import ConnectionPressureMode, WellControlSpec
from bores.wells.resolution.tubing import WellTubing, compute_tubing_head_pressure
from bores.wells.state import ConnectionSample, PhaseValues
from bores.wells.workspace import (
    ConnectionRates,
    PerforationWorkspace,
    accumulate_phase_rates,
    correct_well_indices_for_non_darcy,
    get_connection_rates,
)

__all__ = [
    "PRODUCER_RATE_MODE_PHASES",
    "RATE_QUANTITY_PHASES",
    "bisect_bhp",
    "compute_phase_rates",
    "get_default_pressure_bracket",
    "phase_mask",
    "solve_connection_pressures_and_rates",
    "solve_injector_bhp_mode",
    "solve_injector_rate_mode",
    "solve_producer_bhp_mode",
    "solve_producer_rate_mode",
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


def get_default_pressure_bracket(
    connection_samples: typing.Sequence[ConnectionSample],
    *,
    is_injector: bool,
    control_spec: WellControlSpec,
) -> tuple[Number, Number]:
    """
    Builds a default `(min_pressure, max_pressure)` BHP bisection bracket.

    A producer's BHP can't usefully exceed the highest connected-cell
    pressure as there's no drawdown left to give a rate above that. An
    injector's upper bound is reservoir pressure scaled by
    `control_spec.injector_bhp_bracket_multiplier`, giving room above
    reservoir pressure to actually inject.

    :param connection_samples: Reservoir samples for every connection.
    :param is_injector: Selects which bracket shape to use.
    :param control_spec: Supplies `producer_bhp_floor`/`injector_bhp_bracket_multiplier`.
    :returns: `(min_pressure, max_pressure)`.
    """
    cell_pressures = [sample.pressure for sample in connection_samples]
    if is_injector:
        return min(cell_pressures), control_spec.injector_bhp_bracket_multiplier * max(
            cell_pressures
        )
    return control_spec.producer_bhp_floor, max(cell_pressures)


def solve_connection_pressures_and_rates(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    reference_pressure: Number,
    relevant_phases: PhaseValues,
    is_injector: bool,
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
    `control_spec.max_fixed_point_iterations` is reached (best-effort).
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
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
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
        `control_spec.connection_pressure_mode` is `HYDRAULIC`.
    """
    wellbore = tubing.wellbore
    pressure_mode = control_spec.connection_pressure_mode
    if wellbore is None and pressure_mode == ConnectionPressureMode.HYDRAULIC:
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
    uniform_connection_pressure = (
        wellbore is None and pressure_mode == ConnectionPressureMode.UNIFORM_BHP
    )
    hydrostatic_connection_pressure = (
        wellbore is None and pressure_mode == ConnectionPressureMode.HYDROSTATIC
    )

    zero_rates = PhaseValues(oil=0.0, water=0.0, gas=0.0)
    if uniform_connection_pressure:
        connection_pressures = workspace.connection_pressures
        connection_pressures[:] = reference_pressure
    elif hydrostatic_connection_pressure:
        connection_pressures = compute_hydrostatic_connection_pressures(
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            representative_depths=workspace.representative_depths,
            connection_samples=connection_samples,
            connection_rates=ConnectionRates(
                oil=np.zeros_like(workspace.connection_oil_rates),
                water=np.zeros_like(workspace.connection_water_rates),
                gas=np.zeros_like(workspace.connection_gas_rates),
            ),
            unit_system=control_spec.unit_system,
            out=workspace.connection_pressures,
        )
    else:
        connection_pressures = compute_perforation_pressures_from_fluids(
            wellbore=typing.cast(WellBoreModel, wellbore),
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            connection_oil_rates=np.zeros_like(workspace.connection_oil_rates),
            connection_water_rates=np.zeros_like(workspace.connection_water_rates),
            connection_gas_rates=np.zeros_like(workspace.connection_gas_rates),
            reference_measured_depth=workspace.reference_measured_depth,
            representative_depths=workspace.representative_depths,
            representative_measured_depths=workspace.representative_measured_depths,
            inclinations_from_vertical=workspace.inclinations_from_vertical,
            connection_fluids=workspace.connection_fluids,
            is_injector=is_injector,
            out=workspace.connection_pressures,
        )

    total_rate = 0.0
    previous_total_rate = 0.0
    phase_rates = zero_rates
    surface_phase_rates = zero_rates

    relevant_oil = relevant_phases.oil > 0.0
    relevant_water = relevant_phases.water > 0.0
    relevant_gas = relevant_phases.gas > 0.0

    for _ in range(control_spec.max_fixed_point_iterations):
        correct_well_indices_for_non_darcy(
            static_well_indices=workspace.static_well_indices,
            connection_conductivities=workspace.connection_conductivities,
            connection_gas_rates=workspace.connection_gas_rates,
            d_factor=workspace.d_factor,
            out_well_indices=workspace.well_indices,
        )
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_phase_rates(
            connection_pressures=connection_pressures,
            well_indices=workspace.well_indices,
            reservoir_pressures=workspace.reservoir_pressures,
            oil_mobilities=workspace.oil_mobilities,
            water_mobilities=workspace.water_mobilities,
            gas_mobilities=workspace.gas_mobilities,
            oil_formation_volume_factors=workspace.oil_formation_volume_factors,
            water_formation_volume_factors=workspace.water_formation_volume_factors,
            gas_formation_volume_factors=workspace.gas_formation_volume_factors,
            relevant_oil=relevant_oil or not is_injector,
            relevant_water=relevant_water or not is_injector,
            relevant_gas=relevant_gas or not is_injector,
            is_injector=is_injector,
            radial_flow_constant=control_spec.radial_flow_constant,
            out_connection_oil_rates=workspace.connection_oil_rates,
            out_connection_water_rates=workspace.connection_water_rates,
            out_connection_gas_rates=workspace.connection_gas_rates,
        )
        phase_rates = PhaseValues(
            oil=oil_rate if relevant_oil else 0.0,
            water=water_rate if relevant_water else 0.0,
            gas=gas_rate if relevant_gas else 0.0,
        )
        surface_phase_rates = PhaseValues(
            oil=surface_oil_rate if relevant_oil else 0.0,
            water=surface_water_rate if relevant_water else 0.0,
            gas=surface_gas_rate if relevant_gas else 0.0,
        )
        new_total_rate = phase_rates.oil + phase_rates.water + phase_rates.gas
        if hydrostatic_connection_pressure:
            connection_pressures = compute_hydrostatic_connection_pressures(
                reference_depth=reference_depth,
                reference_pressure=reference_pressure,
                representative_depths=workspace.representative_depths,
                connection_samples=connection_samples,
                connection_rates=get_connection_rates(workspace=workspace),
                unit_system=control_spec.unit_system,
                out=workspace.connection_pressures,
            )
        elif not uniform_connection_pressure:
            connection_pressures = compute_perforation_pressures_from_fluids(
                wellbore=typing.cast(WellBoreModel, wellbore),
                reference_depth=reference_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                reference_measured_depth=workspace.reference_measured_depth,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                out=workspace.connection_pressures,
            )
        if abs(new_total_rate - total_rate) <= control_spec.rate_convergence_tolerance * max(
            abs(new_total_rate), 1.0
        ):
            total_rate = new_total_rate
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate
    else:
        raise WellConvergenceError(
            "The connection pressures and rates did not converge in "
            f"{control_spec.max_fixed_point_iterations} iterations at a reference pressure of "
            f"{reference_pressure}: on the last iteration the total rate went from "
            f"{previous_total_rate} to {total_rate}, against a relative tolerance of "
            f"{control_spec.rate_convergence_tolerance}. A rate high enough that friction or "
            "head takes the connection pressures past reservoir pressure can cause this."
        )

    return connection_pressures, phase_rates, surface_phase_rates


def compute_phase_rates(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    reference_pressure: Number,
    relevant_phases: PhaseValues,
    is_injector: bool,
    control_spec: WellControlSpec,
) -> tuple[NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    `solve_connection_pressures_and_rates` at a fixed BHP.

    For a producer, callers pass `relevant_phases=ALL_PHASES` - a producer
    can legitimately flow oil, water, and gas simultaneously. For an
    injector, `relevant_phases` must be `phase_mask(injected_phase)` only -
    an injector displaces reservoir fluid rather than having an
    IPR-driven rate of its own for the other two phases.

    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param reference_pressure: BHP to report rates at.
    :param relevant_phases: Mask of which phases to compute a real rate for.
    :param is_injector: Selects the drawdown sign convention.
    :param control_spec: Forwarded to `solve_connection_pressures_and_rates`.
    :returns: `(connection_pressures, phase_rates, surface_phase_rates)` -
        flowing pressure at each connection, reservoir-condition phase
        rates, and surface-condition phase rates.
    """
    return solve_connection_pressures_and_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=reference_pressure,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        control_spec=control_spec,
    )


def bisect_bhp(
    *,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    relevant_phases: PhaseValues,
    is_injector: bool,
    target: Number,
    min_pressure: Number,
    max_pressure: Number,
    control_spec: WellControlSpec,
    metric: typing.Literal["rate", "thp"] = "rate",
    target_rate_condition: typing.Literal["surface", "reservoir"] = "surface",
    surface_fluid_properties: SurfaceFluidProperties | None = None,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues]:
    """
    Bisects `reference_pressure` in `[min_pressure, max_pressure]` until
    the resulting metric (`relevant_phases` total rate, or THP) matches `target`.

    Both metrics are monotonic in BHP, in a direction that depends on the
    metric and (for rate) the well type. For `metric="rate"`, lower BHP means
    a higher producer rate, higher BHP means a higher injector rate;
    For `metric="thp"`, higher BHP means higher THP, for both well types.

    `min_pressure`/`max_pressure` must bracket the target. Callers should pick
    these via `get_default_pressure_bracket` (or their own). Best-effort
    (no exception) if `target` isn't achievable within the bracket.
    Returns the closest bound reached after `control_spec.max_bisection_iterations`.

    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param relevant_phases: Mask the `"rate"` metric sums over.
    :param is_injector: Selects the drawdown sign convention and the
        direction `"rate"` moves in as BHP changes.
    :param target: Value the chosen metric is bisected toward.
    :param min_pressure: Lower bracket bound.
    :param max_pressure: Upper bracket bound.
    :param control_spec: Supplies the iteration cap and convergence tolerance.
    :param metric: `"rate"` (default) matches `relevant_phases`' total
        rate against `target`; `"thp"` matches tubing head pressure
        against `target` instead (requires `surface_fluid_properties`).
    :param target_rate_condition: `"surface"` (default) bisects against
        the surface-condition total; `"reservoir"` bisects against the
        raw reservoir-condition total instead. Ignored when `metric="thp"`.
    :param surface_fluid_properties: Required if `metric="thp"`.
    :returns: `(bhp, connection_pressures, phase_rates)`, `phase_rates`
        always reservoir-condition regardless of `target_rate_condition`.
    :raises ValidationError: If `metric="thp"` and `surface_fluid_properties` is `None`.
    """
    if metric == "thp" and surface_fluid_properties is None and tubing.vfp_table is None:
        raise ValidationError("bisect_bhp(..., metric='thp') requires `surface_fluid_properties`.")

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
            connection_samples=connection_samples,
            reference_pressure=average_pressure,
            relevant_phases=relevant_phases,
            is_injector=is_injector,
            control_spec=control_spec,
        )

        if metric == "rate":
            target_rates = (
                phase_rates if target_rate_condition == "reservoir" else surface_phase_rates
            )
            value = target_rates.oil + target_rates.water + target_rates.gas
            increasing_with_bhp = is_injector
        else:
            value = compute_tubing_head_pressure(
                tubing=tubing,
                reference_depth=reference_depth,
                reference_pressure=average_pressure,
                phase_rates=phase_rates,
                surface_phase_rates=surface_phase_rates,
                surface_fluid_properties=surface_fluid_properties,
                is_injector=is_injector,
                connection_rates=get_connection_rates(workspace=workspace),
            )
            increasing_with_bhp = True

        if abs(value - target) <= control_spec.rate_convergence_tolerance * max(abs(target), 1.0):
            break

        if (value < target) == increasing_with_bhp:
            low = average_pressure
        else:
            high = average_pressure

    assert pressures is not None
    return average_pressure, pressures, phase_rates


def solve_producer_rate_mode(
    *,
    control_mode: Integer,
    target_rate: Number,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
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
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param control_spec: Solver tunables.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)`,
        reservoir- and surface-condition rates for every phase.
    """
    relevant_phases = PRODUCER_RATE_MODE_PHASES[control_mode]
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples, is_injector=False, control_spec=control_spec
    )
    bhp, _, _ = bisect_bhp(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
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
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        control_spec=control_spec,
    )
    return bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_producer_bhp_mode(
    *,
    target_bhp: Number,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    control_spec: WellControlSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Resolves a producer held at a fixed BHP; rates are the output.

    Nominal resolution only. See `solve_producer_rate_mode`.
    """
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=target_bhp,
        relevant_phases=ALL_PHASES,
        is_injector=False,
        control_spec=control_spec,
    )
    return target_bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_injector_rate_mode(
    *,
    control_mode: Integer,
    target_rate: Number,
    injected_phase: Integer,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    control_spec: WellControlSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Injector analogue of `solve_producer_rate_mode` - `injected_phase`
    selects the single phase being allocated.

    Nominal resolution only. See `solve_producer_rate_mode`.

    :param control_mode: `InjectorControlModeTag` value.
    :param target_rate: The rate target.
    :param injected_phase: `FluidPhaseTag` value of the injected phase.
    :param tubing: Hydraulics correlation for this well.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: This well's `PerforationWorkspace`.
    :param connection_samples: Reservoir samples, same order as `workspace`'s arrays.
    :param control_spec: Solver tunables.
    :returns: `(bhp, connection_pressures, phase_rates, surface_phase_rates)` -
        reservoir- and surface-condition rates, `injected_phase` only nonzero.
    """
    relevant_phases = phase_mask(injected_phase)
    min_pressure, max_pressure = get_default_pressure_bracket(
        connection_samples, is_injector=True, control_spec=control_spec
    )
    bhp, _, _ = bisect_bhp(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
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
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=bhp,
        relevant_phases=relevant_phases,
        is_injector=True,
        control_spec=control_spec,
    )
    return bhp, connection_pressures, phase_rates, surface_phase_rates


def solve_injector_bhp_mode(
    *,
    target_bhp: Number,
    injected_phase: Integer,
    tubing: WellTubing,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    connection_samples: typing.Sequence[ConnectionSample],
    control_spec: WellControlSpec,
) -> tuple[Number, NumberArray[OneDimension], PhaseValues, PhaseValues]:
    """
    Injector analogue of `solve_producer_bhp_mode`.

    Nominal resolution only. See `solve_producer_rate_mode`.
    """
    relevant_phases = phase_mask(injected_phase)
    connection_pressures, phase_rates, surface_phase_rates = compute_phase_rates(
        tubing=tubing,
        reference_depth=reference_depth,
        workspace=workspace,
        connection_samples=connection_samples,
        reference_pressure=target_bhp,
        relevant_phases=relevant_phases,
        is_injector=True,
        control_spec=control_spec,
    )
    return target_bhp, connection_pressures, phase_rates, surface_phase_rates
