"""Turns a resolved compiled well structure back into rich, readable objects."""

import math

import attrs

from bores.types import FluidPhase, UnitSystem
from bores.wells.base import AnyPerforation, CompletionStatus, Well, Wells, WellStatus, WellType
from bores.wells.compile import (
    UNSET_INT,
    CompiledGroupControls,
    CompiledPerforations,
    CompiledWellControls,
    CompiledWellSystem,
    EconomicQuantityTag,
    FluidPhaseTag,
    GroupInjectorControlModeTag,
    GroupKind,
    GroupProducerControlModeTag,
    InjectorControlModeTag,
    LimitKind,
    ProducerControlModeTag,
    RateQuantityTag,
    WellKind,
    WorkoverActionTag,
)
from bores.wells.controls import (
    BHPLimit,
    EconomicLimit,
    EconomicQuantity,
    InjectorControl,
    InjectorControlMode,
    Limit,
    ProducerControl,
    ProducerControlMode,
    RateLimit,
    RateQuantity,
    THPLimit,
    WellControl,
    WellControls,
    WorkoverAction,
)
from bores.wells.groups import (
    GroupControl,
    GroupControls,
    GroupInjectorControlMode,
    GroupProducerControlMode,
)
from bores.wells.model import WellSystem
from bores.wells.resolution.compile import CompiledWellResolution
from bores.wells.states import PerforationState, PhaseValues, WellsStates, WellState

__all__ = [
    "build_wells_states",
    "decompile_control",
    "decompile_group_control",
    "decompile_group_controls",
    "decompile_limit",
    "decompile_perforation",
    "decompile_perforations",
    "decompile_well",
    "decompile_well_control",
    "decompile_well_controls",
    "decompile_well_perforations",
    "decompile_well_system",
    "decompile_wells",
]

PRODUCER_MODE_FROM_TAG = {
    ProducerControlModeTag.OIL_RATE: ProducerControlMode.OIL_RATE,
    ProducerControlModeTag.WATER_RATE: ProducerControlMode.WATER_RATE,
    ProducerControlModeTag.GAS_RATE: ProducerControlMode.GAS_RATE,
    ProducerControlModeTag.LIQUID_RATE: ProducerControlMode.LIQUID_RATE,
    ProducerControlModeTag.RESERVOIR_VOLUME_RATE: ProducerControlMode.RESERVOIR_VOLUME_RATE,
    ProducerControlModeTag.BHP: ProducerControlMode.BHP,
    ProducerControlModeTag.THP: ProducerControlMode.THP,
    ProducerControlModeTag.GROUP: ProducerControlMode.GROUP,
}
INJECTOR_MODE_FROM_TAG = {
    InjectorControlModeTag.RATE: InjectorControlMode.RATE,
    InjectorControlModeTag.RESERVOIR_VOLUME_RATE: InjectorControlMode.RESERVOIR_VOLUME_RATE,
    InjectorControlModeTag.BHP: InjectorControlMode.BHP,
    InjectorControlModeTag.THP: InjectorControlMode.THP,
    InjectorControlModeTag.GROUP: InjectorControlMode.GROUP,
}
GROUP_PRODUCER_MODE_FROM_TAG = {
    GroupProducerControlModeTag.OIL_RATE: GroupProducerControlMode.OIL_RATE,
    GroupProducerControlModeTag.WATER_RATE: GroupProducerControlMode.WATER_RATE,
    GroupProducerControlModeTag.GAS_RATE: GroupProducerControlMode.GAS_RATE,
    GroupProducerControlModeTag.LIQUID_RATE: GroupProducerControlMode.LIQUID_RATE,
    GroupProducerControlModeTag.RESERVOIR_VOLUME_RATE: (
        GroupProducerControlMode.RESERVOIR_VOLUME_RATE
    ),
    GroupProducerControlModeTag.FIELD: GroupProducerControlMode.FIELD,
    GroupProducerControlModeTag.NONE: GroupProducerControlMode.NONE,
}
GROUP_INJECTOR_MODE_FROM_TAG = {
    GroupInjectorControlModeTag.RATE: GroupInjectorControlMode.RATE,
    GroupInjectorControlModeTag.RESERVOIR_VOLUME_RATE: (
        GroupInjectorControlMode.RESERVOIR_VOLUME_RATE
    ),
    GroupInjectorControlModeTag.VOIDAGE_REPLACEMENT: GroupInjectorControlMode.VOIDAGE_REPLACEMENT,
    GroupInjectorControlModeTag.REINJECTION: GroupInjectorControlMode.REINJECTION,
    GroupInjectorControlModeTag.FIELD: GroupInjectorControlMode.FIELD,
}
RATE_QUANTITY_FROM_TAG = {
    RateQuantityTag.OIL: RateQuantity.OIL,
    RateQuantityTag.WATER: RateQuantity.WATER,
    RateQuantityTag.GAS: RateQuantity.GAS,
    RateQuantityTag.LIQUID: RateQuantity.LIQUID,
    RateQuantityTag.RESERVOIR: RateQuantity.RESERVOIR,
}
ECONOMIC_QUANTITY_FROM_TAG = {
    EconomicQuantityTag.WATER_CUT: EconomicQuantity.WATER_CUT,
    EconomicQuantityTag.GOR: EconomicQuantity.GOR,
    EconomicQuantityTag.WATER_GAS_RATIO: EconomicQuantity.WATER_GAS_RATIO,
    EconomicQuantityTag.OIL_RATE: EconomicQuantity.OIL_RATE,
    EconomicQuantityTag.GAS_RATE: EconomicQuantity.GAS_RATE,
}
WORKOVER_ACTION_FROM_TAG = {
    WorkoverActionTag.WELL: WorkoverAction.WELL,
    WorkoverActionTag.PLUG: WorkoverAction.PLUG,
    WorkoverActionTag.CON: WorkoverAction.CON,
    WorkoverActionTag.PLUS_CON: WorkoverAction.PLUS_CON,
}
FLUID_PHASE_FROM_TAG = {
    FluidPhaseTag.OIL: FluidPhase.OIL,
    FluidPhaseTag.WATER: FluidPhase.WATER,
    FluidPhaseTag.GAS: FluidPhase.GAS,
}
WELL_TYPE_FROM_KIND = {
    WellKind.PRODUCER: WellType.PRODUCER,
    WellKind.INJECTOR: WellType.INJECTOR,
}


def get_well_status(tag: int) -> WellStatus:
    """
    Converts a compiled well-status tag back to `WellStatus`. Inverse of
    `bores.wells.compile.get_well_status_tag`.

    :param tag: `1` for `ACTIVE`, `0` for `PENDING`.
    :returns: The matching `WellStatus`.
    """
    return WellStatus.ACTIVE if tag else WellStatus.PENDING


def get_completion_status(tag: int) -> CompletionStatus:
    """
    Converts a compiled completion-status tag back to `CompletionStatus`.
    Inverse of `bores.wells.compile.get_completion_status_tag`.

    :param tag: `1` for `OPEN`, `0` for `SHUT`.
    :returns: The matching `CompletionStatus`.
    """
    return CompletionStatus.OPEN if tag else CompletionStatus.SHUT


def none_if_nan(value: float) -> float | None:
    """
    Returns `value`, or `None` if it's `NaN`

    :param value: A possibly-`NaN` float from a compiled array.
    :returns: `value`, or `None` if it's `NaN`.
    """
    return None if math.isnan(value) else value


def decompile_perforation(
    original: AnyPerforation, perforations: CompiledPerforations, row: int
) -> AnyPerforation:
    """
    Rebuilds one rich perforation from a single compiled connection row.

    `original` supplies every field the compiled layer doesn't carry (the
    depth range, or the measured-depth range and trajectory reference for
    an `MDPerforation`). `status`, `schedule_status`, `skin`,
    `wellbore_radius`, and `saturation_region` are all overridden from the
    compiled row instead, so this reflects any in-place patch applied to
    the compiled layer since compile time (a `WELOPEN` event, a `WPIMULT`
    reissue) rather than trusting a possibly-stale original.

    :param original: The source rich `Perforation`/`MDPerforation`,
        matched via `CompiledPerforations.perforation_indices`.
    :param perforations: `CompiledPerforations` for the whole system.
    :param row: The compiled connection row to read current state from.
    :returns: `original`, evolved with the compiled row's current state.
    """
    saturation_region = perforations.saturation_regions[row]
    return attrs.evolve(
        original,
        status=get_completion_status(perforations.completion_statuses[row]),
        schedule_status=get_well_status(perforations.schedule_statuses[row]),
        skin=perforations.skins[row],
        wellbore_radius=perforations.wellbore_radii[row],
        saturation_region=None if saturation_region == UNSET_INT else int(saturation_region),
    )


def decompile_perforations(
    wells: Wells, well_name: str, perforations: CompiledPerforations, well_row: int
) -> tuple[AnyPerforation, ...]:
    """
    Rebuilds one well's per-connection rich perforations from its rows of
    `CompiledPerforations`.

    Returns one entry per compiled connection row for this well, not one
    per rich perforation - a rich perforation whose trajectory crosses
    several grid cells produces several consecutive rows here, each
    rebuilt from that row's own current state (see `decompile_perforation`).
    Matched through `CompiledPerforations.perforation_indices`, not row
    position, since a well with any multi-cell perforation would
    otherwise misalign. Use `decompile_well_perforations` instead when
    rebuilding a standalone `Well`, which wants one entry per original
    completion, not one per grid cell.

    :param wells: The original rich `Wells` this system was compiled
        from. The compiled layer keeps no reference back to these, so
        this is required to get anything but positions and cell indices.
    :param well_name: This well's name, to look it up in `wells`.
    :param perforations: `CompiledPerforations` for the whole system.
    :param well_row: This well's row, positionally aligned with
        `CompiledWellSystem.names`.
    :returns: One rich `AnyPerforation` per compiled connection row for
        this well, in `CompiledPerforations` row order.
    """
    row_start = perforations.well_offsets[well_row]
    row_end = perforations.well_offsets[well_row + 1]
    rich_perforations = wells[well_name].perforations
    return tuple(
        decompile_perforation(
            rich_perforations[perforations.perforation_indices[row]], perforations, row
        )
        for row in range(row_start, row_end)
    )


def decompile_well_perforations(
    wells: Wells, well_name: str, perforations: CompiledPerforations, well_row: int
) -> tuple[AnyPerforation, ...]:
    """
    Rebuilds one well's rich perforations at completion granularity - one
    entry per original `Well.perforations` entry, not one per grid cell -
    for reconstructing a standalone `Well`.

    A completion whose trajectory or depth range crosses several grid
    cells produced several consecutive rows in `CompiledPerforations`;
    those rows can end up patched independently of each other (a
    per-connection `WELOPEN` naming specific cells within one wide
    completion). This collapses each such group back to one
    representative perforation using its first row, so it cannot show
    that kind of within-completion divergence. Use `decompile_perforations`
    instead when that level of detail matters - it matches
    `PerforationState`, one entry per connection.

    :param wells: The original rich `Wells` this system was compiled from.
    :param well_name: This well's name, to look it up in `wells`.
    :param perforations: `CompiledPerforations` for the whole system.
    :param well_row: This well's row.
    :returns: One rich `AnyPerforation` per original completion, in the
        same order as the source `Well.perforations`.
    """
    row_start = perforations.well_offsets[well_row]
    row_end = perforations.well_offsets[well_row + 1]
    rich_perforations = wells[well_name].perforations

    rebuilt = []
    row = row_start
    while row < row_end:
        ordinal = perforations.perforation_indices[row]
        rebuilt.append(decompile_perforation(rich_perforations[ordinal], perforations, row))
        row += 1
        while row < row_end and perforations.perforation_indices[row] == ordinal:
            row += 1
    return tuple(rebuilt)


def decompile_group_control(
    group_controls: CompiledGroupControls, row: int, unit_system: UnitSystem
) -> GroupControl:
    """
    Rebuilds one rich `GroupControl` from a single row of `CompiledGroupControls`.

    :param group_controls: The compiled group controls.
    :param row: The row to rebuild.
    :param unit_system: Unit system to tag the rebuilt control with.
    :returns: The rebuilt `GroupControl`.
    """
    target_rate = none_if_nan(group_controls.target_rates[row])
    injected_phase_tag = group_controls.injected_phases[row]

    if group_controls.group_kinds[row] == GroupKind.INJECTOR:
        return GroupControl(
            mode=GROUP_INJECTOR_MODE_FROM_TAG[group_controls.control_modes[row]],
            target_rate=target_rate,
            injected_phase=(
                FLUID_PHASE_FROM_TAG[injected_phase_tag]
                if injected_phase_tag != UNSET_INT
                else None
            ),
            unit_system=unit_system,
        )
    return GroupControl(
        mode=GROUP_PRODUCER_MODE_FROM_TAG[group_controls.control_modes[row]],
        target_rate=target_rate,
        unit_system=unit_system,
    )


def decompile_group_controls(
    group_controls: CompiledGroupControls | None, unit_system: UnitSystem
) -> GroupControls | None:
    """
    Rebuilds a rich `GroupControls` from `CompiledGroupControls`.

    Only rebuilds each group's control target, not its resolved
    membership - `CompiledGroupControls.member_well_indices` is a
    solve-time convenience (already-resolved well positions for
    allocation), not something a rich `GroupControl` itself carries. The
    group hierarchy (`WellGroups`) this was compiled against is the
    source of truth for membership, separately from this.

    :param group_controls: `CompiledGroupControls`, or `None` if no group
        has an explicit control.
    :param unit_system: Unit system to tag every rebuilt control with.
    :returns: `GroupControls` keyed by group name, or `None` if
        `group_controls` is `None`.
    """
    if group_controls is None:
        return None
    controls = {
        name: decompile_group_control(group_controls, row, unit_system)
        for row, name in enumerate(group_controls.names)
    }
    return GroupControls(controls=controls, unit_system=unit_system)


def decompile_limit(controls: CompiledWellControls, row: int, unit_system: UnitSystem) -> Limit:
    """
    Rebuilds one rich `Limit` from a single row of `CompiledLimits`.

    :param controls: This well's `CompiledWellControls`, for `.limits`.
    :param row: The row to rebuild.
    :param unit_system: Unit system to tag the rebuilt limit with.
    :returns: The rebuilt `BHPLimit`, `THPLimit`, `RateLimit`, or `EconomicLimit`.
    :raises ValueError: If the row's `LimitKind` isn't recognized.
    """
    limits = controls.limits
    kind = limits.kinds[row]
    min_value = none_if_nan(limits.min_values[row])
    max_value = none_if_nan(limits.max_values[row])

    if kind == LimitKind.BHP:
        return BHPLimit(min_value=min_value, max_value=max_value, unit_system=unit_system)
    if kind == LimitKind.THP:
        return THPLimit(min_value=min_value, max_value=max_value, unit_system=unit_system)
    if kind == LimitKind.RATE:
        assert max_value is not None, "`LimitKind.RATE` `max_value` row cannot be None"
        return RateLimit(
            quantity=RATE_QUANTITY_FROM_TAG[limits.quantities[row]],
            max_value=max_value,
            unit_system=unit_system,
        )
    if kind == LimitKind.ECONOMIC:
        return EconomicLimit(
            quantity=ECONOMIC_QUANTITY_FROM_TAG[limits.quantities[row]],
            min_value=min_value,
            max_value=max_value,
            workover_action=WORKOVER_ACTION_FROM_TAG[limits.workover_actions[row]],
            end_run=bool(limits.end_run_flags[row]),
            unit_system=unit_system,
        )
    raise ValueError(f"Unknown `LimitKind`: {kind!r}.")


def decompile_control(
    controls: CompiledWellControls,
    well_row: int,
    all_limits: tuple[Limit, ...],
    unit_system: UnitSystem,
) -> WellControl:
    """
    Rebuilds one well's rich `ProducerControl`/`InjectorControl` from its
    row in `CompiledWellControls`.

    Read from the compiled row rather than looked up from the well's
    original rich control, since the compiled row is the one that could
    have moved since compile time (e.g, from a group allocation share, a `WELTARG`
    event), and it should be what this reflects.

    :param controls: `CompiledWellControls` for the whole system.
    :param well_row: This well's row.
    :param all_limits: This well's already-rebuilt limits, in `CompiledLimits` row order.
    :param unit_system: Unit system to tag the rebuilt control with.
    :returns: The rebuilt `ProducerControl` or `InjectorControl`.
    """
    target_rate = none_if_nan(controls.target_rates[well_row])
    target_bhp = none_if_nan(controls.target_bhps[well_row])
    target_thp = none_if_nan(controls.target_thps[well_row])
    efficiency_factor = controls.efficiency_factors[well_row]
    guide_rate = none_if_nan(controls.guide_rates[well_row])

    if controls.well_kinds[well_row] == WellKind.INJECTOR:
        return InjectorControl(
            injected_phase=FLUID_PHASE_FROM_TAG[controls.injected_phases[well_row]],
            mode=INJECTOR_MODE_FROM_TAG[controls.control_modes[well_row]],
            target_rate=target_rate,
            target_bhp=target_bhp,
            target_thp=target_thp,
            limits=all_limits,
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    return ProducerControl(
        mode=PRODUCER_MODE_FROM_TAG[controls.control_modes[well_row]],
        target_rate=target_rate,
        target_bhp=target_bhp,
        target_thp=target_thp,
        limits=all_limits,
        efficiency_factor=efficiency_factor,
        guide_rate=guide_rate,
        unit_system=unit_system,
    )


def decompile_well_control(
    controls: CompiledWellControls, well_row: int, unit_system: UnitSystem
) -> WellControl:
    """
    Rebuilds one well's rich `WellControl`, own limits included, from its
    row in `CompiledWellControls`.

    Composes `decompile_limit` for every one of this well's rows in
    `controls.limits`, then `decompile_control` for the control target
    itself.

    :param controls: `CompiledWellControls` for the whole system.
    :param well_row: This well's row.
    :param unit_system: Unit system to tag the rebuilt control and limits with.
    :returns: The rebuilt `ProducerControl` or `InjectorControl`.
    """
    limits_start = controls.limits.well_offsets[well_row]
    limits_end = controls.limits.well_offsets[well_row + 1]
    well_limits = tuple(
        decompile_limit(controls, row, unit_system) for row in range(limits_start, limits_end)
    )
    return decompile_control(controls, well_row, well_limits, unit_system)


def decompile_well_controls(compiled_system: CompiledWellSystem) -> WellControls:
    """
    Rebuilds a rich `WellControls` from every well's row in a `CompiledWellSystem`.

    :param compiled_system: The compiled system to read from.
    :returns: `WellControls` keyed by well name, one entry per row in
        `compiled_system.names`.
    """
    controls = compiled_system.controls
    unit_system = compiled_system.unit_system
    result = {
        well_name: decompile_well_control(controls, well_row, unit_system)
        for well_row, well_name in enumerate(compiled_system.names)
    }
    return WellControls(controls=result, unit_system=unit_system)


def decompile_well(wells: Wells, compiled_system: CompiledWellSystem, well_row: int) -> Well:
    """
    Rebuilds one rich `Well` from its row in a `CompiledWellSystem`.

    `surface_location`, `trajectory`, `preferred_phase`, `group`, and
    `pvt_region` are read straight from the matching original in `wells`,
    since the compiled layer carries no geometry or grouping data at all -
    there's nothing to rebuild them from. `well_type`, `reference_depth`,
    and `schedule_status` come from `compiled_system` directly, and
    `perforations` is rebuilt via `decompile_well_perforations`, so all
    four reflect any in-place patch made since compile time.

    :param wells: The original rich `Wells` this system was compiled from.
    :param compiled_system: The compiled system to read from.
    :param well_row: This well's row, positionally aligned with
        `compiled_system.names`.
    :returns: The rebuilt `Well`.
    """
    well_name = compiled_system.names[well_row]
    original = wells[well_name]
    perforations = decompile_well_perforations(
        wells, well_name, compiled_system.perforations, well_row
    )
    return attrs.evolve(
        original,
        well_type=WELL_TYPE_FROM_KIND[compiled_system.well_kinds[well_row]],
        reference_depth=compiled_system.reference_depths[well_row],
        schedule_status=get_well_status(compiled_system.schedule_statuses[well_row]),
        perforations=perforations,
    )


def decompile_wells(wells: Wells, compiled_system: CompiledWellSystem) -> Wells:
    """
    Rebuilds every well in a `CompiledWellSystem` as rich `Wells`.

    :param wells: The original rich `Wells` this system was compiled from.
    :param compiled_system: The compiled system to read from.
    :returns: `Wells` keyed by name, one entry per row in
        `compiled_system.names`.
    """
    rebuilt = {
        well_name: decompile_well(wells, compiled_system, well_row)
        for well_row, well_name in enumerate(compiled_system.names)
    }
    return Wells(wells=rebuilt, unit_system=compiled_system.unit_system)


def decompile_well_system(
    well_system: WellSystem, compiled_system: CompiledWellSystem
) -> WellSystem:
    """
    Rebuilds a full rich `WellSystem` from a `CompiledWellSystem`.

    `wells`, `well_controls`, and `group_controls` are rebuilt entirely
    from `compiled_system`, so they reflect any in-place patch made to it
    since compile time. `default_wellbore`, `wellbore_overrides`,
    `groups`, and `resolver_spec` are carried over from `well_system`
    unchanged - none of these are part of the compiled representation.
    Hydraulics correlation choice is dispatched by name at call time, not
    stored per well, and the group hierarchy and resolver configuration
    aren't hot-path data at all, so there's nothing in `compiled_system`
    to rebuild them from.

    :param well_system: The original rich `WellSystem` `compiled_system`
        was compiled from.
    :param compiled_system: The compiled system to read `wells`,
        `well_controls`, and `group_controls` from.
    :returns: A new `WellSystem` reflecting `compiled_system`'s current state.
    """
    return attrs.evolve(
        well_system,
        wells=decompile_wells(well_system.wells, compiled_system),
        well_controls=decompile_well_controls(compiled_system),
        group_controls=decompile_group_controls(
            compiled_system.group_controls, compiled_system.unit_system
        ),
    )


def build_wells_states(
    wells: Wells,
    compiled_system: CompiledWellSystem,
    resolution: CompiledWellResolution,
) -> WellsStates:
    """
    Builds `WellsStates` from a resolved `CompiledWellResolution`.

    Only covers wells actually resolved this pass. A well whose
    `WellStatus` is still `PENDING` (its BHP is left `NaN` by
    `resolve_control`) is skipped rather than reported with meaningless
    values.

    Composes `decompile_perforations`, `decompile_limit`, and
    `decompile_well_control` rather than rebuilding any of them inline,
    so each piece can be reused or tested on its own.

    :param wells: The original rich `Wells` this system was compiled
        from. This supplies each `PerforationState.perforation`, which the
        compiled layer doesn't retain a reference to.
    :param compiled_system: The system `resolution` was resolved against.
    :param resolution: A `CompiledWellResolution` from a completed resolve pass.
    :returns: One `WellState` per resolved well, keyed by well name.
    """
    controls = compiled_system.controls
    perforations = compiled_system.perforations
    unit_system = compiled_system.unit_system

    states: dict[str, WellState] = {}
    for well_row, well_name in enumerate(compiled_system.names):
        bhp = resolution.bhps[well_row]
        if math.isnan(bhp):
            continue  # not resolved this pass (PENDING, or UNSET control)

        row_start = perforations.well_offsets[well_row]
        rich_perforations = decompile_perforations(wells, well_name, perforations, well_row)

        perforation_states = []
        for row in range(row_start, perforations.well_offsets[well_row + 1]):
            pressure = resolution.connection_pressures[row]
            if math.isnan(pressure):
                continue  # this connection wasn't active this pass (shut or pending)

            perforation_states.append(
                PerforationState(
                    perforation=rich_perforations[row - row_start],
                    cell_index=int(perforations.cell_indices[row]),
                    flowing_pressure=pressure,
                    phase_rates=PhaseValues(
                        oil=resolution.connection_oil_rates[row],
                        water=resolution.connection_water_rates[row],
                        gas=resolution.connection_gas_rates[row],
                    ),
                    unit_system=unit_system,
                )
            )

        active_limit_row = resolution.active_limit_rows[well_row]
        active_limit = (
            None
            if active_limit_row == UNSET_INT
            else decompile_limit(controls, active_limit_row, unit_system)
        )

        states[well_name] = WellState(
            well_name=well_name,
            is_open=not bool(resolution.economic_shutins[well_row]),
            active_control=decompile_well_control(controls, well_row, unit_system),
            bhp=bhp,
            perforation_states=tuple(perforation_states),
            phase_rates=PhaseValues(
                oil=resolution.oil_rates[well_row],
                water=resolution.water_rates[well_row],
                gas=resolution.gas_rates[well_row],
            ),
            surface_phase_rates=PhaseValues(
                oil=resolution.surface_oil_rates[well_row],
                water=resolution.surface_water_rates[well_row],
                gas=resolution.surface_gas_rates[well_row],
            ),
            active_limit=active_limit,
            thp=none_if_nan(resolution.thps[well_row]),
            unit_system=unit_system,
        )

    return WellsStates(states=states, unit_system=unit_system)
