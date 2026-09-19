"""
`GECON` group-level economic-limit enforcement.

Checked after a group's member wells have already been resolved for the
current timestep. Every action this module takes against a group's
target or membership is followed by reallocating (`allocate_group_targets`)
and actually re-resolving (`resolve_well_control`) every affected member
still open, before rechecking the group's aggregate - never leaves a
stale rate behind for the caller to reconcile.
"""

import math
import typing

import numpy as np

from bores.errors import StopSimulation, ValidationError
from bores.types import Integer
from bores.wells.compile import (
    UNSET_INT,
    CompiledWellSystem,
    GroupKind,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
    WorkoverActionTag,
)
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellBoreModel
from bores.wells.resolution.engine import resolve_well_control
from bores.wells.resolution.groups.allocation import allocate_group_targets
from bores.wells.resolution.limits import check_economic_violation
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.states import ConnectionSample, PhaseValues
from bores.wells.workspace import WellsWorkspace

__all__ = ["GroupEconomicLimitOutcome", "enforce_group_economic_limits"]


class GroupEconomicLimitOutcome(typing.NamedTuple):
    """Result of one `enforce_group_economic_limits` call."""

    satisfied: bool
    """Whether every one of this group's `GECON` limits is satisfied when this call returns."""

    shut_wells: tuple[str, ...]
    """Names of member wells shut this call, in the order they were shut."""

    rate_cutback_applied: bool
    """Whether this call cut the group's own `target_rate` (`workover_action` was `RATE`)."""

    reallocated_wells: tuple[str, ...]
    """
    Names of `GRUP`-mode member wells whose target was recomputed and who
    were actually re-resolved this call, following a shut-in or a rate cutback.
    """


def aggregate_member_rates(
    *, workspace: WellsWorkspace, open_member_wells: typing.Sequence[Integer]
) -> PhaseValues:
    """
    Sums already-resolved reservoir-condition rates over `open_member_wells`.

    :param workspace: Supplies each member well's own resolved rates.
    :param open_member_wells: Rows to sum over.
    :returns: The group's current aggregate `PhaseValues`.
    """
    oil = sum(workspace.oil_rates[i] for i in open_member_wells)
    water = sum(workspace.water_rates[i] for i in open_member_wells)
    gas = sum(workspace.gas_rates[i] for i in open_member_wells)
    return PhaseValues(oil=oil, water=water, gas=gas)


def shut_in_member_well(
    *, well_system: CompiledWellSystem, workspace: WellsWorkspace, well_row: Integer
) -> None:
    """
    Zeroes one member well's rates in place, well-level and per-connection,
    mirroring `resolution.engine.resolve_well_control`'s own economic
    shut-in zeroing exactly.

    Leaves `workspace.active_limit_rows[well_row]` untouched, since that
    field indexes this well's own `CompiledLimits` rows, a different
    space from the group's `CompiledGroupLimits` row that actually
    triggered this shut-in.

    :param well_system: Supplies `.perforations.well_offsets` for this well's row range.
    :param workspace: Updated in place.
    :param well_row: The member well being shut in.
    """
    workspace.economic_shutins[well_row] = 1
    workspace.oil_rates[well_row] = 0.0
    workspace.water_rates[well_row] = 0.0
    workspace.gas_rates[well_row] = 0.0
    workspace.surface_oil_rates[well_row] = 0.0
    workspace.surface_water_rates[well_row] = 0.0
    workspace.surface_gas_rates[well_row] = 0.0

    row_start = well_system.perforations.well_offsets[well_row]
    row_end = well_system.perforations.well_offsets[well_row + 1]
    resolved_mask = ~np.isnan(workspace.connection_pressures[row_start:row_end])
    workspace.connection_oil_rates[row_start:row_end][resolved_mask] = 0.0
    workspace.connection_water_rates[row_start:row_end][resolved_mask] = 0.0
    workspace.connection_gas_rates[row_start:row_end][resolved_mask] = 0.0


def reallocate_and_reresolve(
    *,
    group_name: str,
    well_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
    open_members: typing.Sequence[Integer],
    grup_mode_tag: Integer,
    grup_member_rows: typing.Sequence[Integer],
    get_wellbore: typing.Callable[[Integer], WellBoreModel],
    get_connection_samples: typing.Callable[[Integer], typing.Sequence[ConnectionSample]],
    get_surface_fluid_properties: typing.Callable[[Integer], SurfaceFluidProperties | None],
) -> tuple[str, ...]:
    """
    Reallocates `group_name`'s current target across `grup_member_rows`
    and actually re-resolves every one still open, in place, so the
    group's aggregate reflects real, freshly resolved rates on the next
    check, not a stale pre-reallocation value.

    `allocate_group_targets` only picks up a member currently in `GRUP`
    mode, but converts it to a concrete mode as part of allocating it.
    So a member already reallocated by an earlier call this pass would
    be silently skipped on a later one. Restoring `grup_member_rows`'
    still-open rows to `grup_mode_tag` immediately before calling it
    keeps every one of this group's own members reallocatable on every
    pass, for as long as this function keeps calling this helper.

    :param open_members: Rows currently eligible to be resolved (open,
        matching well kind, not economically shut).
    :param grup_mode_tag: This group's own `GRUP`-equivalent control mode
        tag (producer or injector, matching `group_name`'s own kind).
    :param grup_member_rows: Rows that were in `GRUP` mode when
        `enforce_group_economic_limits` was first called for this group,
        the members this group's own reallocation is meant to keep
        covering across every pass this call makes.
    :param get_wellbore: Given a well row, its hydraulics correlation.
    :param get_connection_samples: Given a well row, its active, open
        connections' current reservoir samples.
    :param get_surface_fluid_properties: Given a well row, its surface
        fluid properties, or `None` if it has no THP control/limit to check.
    :returns: Names of the members actually re-resolved.
    """
    open_member_set = set(open_members)
    for well_row in grup_member_rows:
        if well_row in open_member_set:
            well_system.controls.control_modes[well_row] = grup_mode_tag

    reallocated = allocate_group_targets(group_name, well_system, workspace)
    reresolved = [
        name for name in reallocated if well_system.well_row(name=name) in open_member_set
    ]
    for name in reresolved:
        well_row = well_system.well_row(name=name)
        resolve_well_control(
            compiled_system=well_system,
            well_row=well_row,
            wellbore=get_wellbore(well_row),
            connection_samples=get_connection_samples(well_row),
            workspace=workspace,
            control_spec=control_spec,
            surface_fluid_properties=get_surface_fluid_properties(well_row),
        )
    return tuple(reresolved)


def enforce_group_economic_limits(
    *,
    group_name: str,
    well_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
    get_wellbore: typing.Callable[[Integer], WellBoreModel],
    get_connection_samples: typing.Callable[[Integer], typing.Sequence[ConnectionSample]],
    get_surface_fluid_properties: typing.Callable[[Integer], SurfaceFluidProperties | None]
    | None = None,
) -> GroupEconomicLimitOutcome:
    """
    Checks `group_name`'s own `GECON` limits against its member wells'
    resolved rates, and drives it to a satisfied state.

    For `WELL`/`PLUG`/`CON`/`PLUS_CON`, shuts in the eligible open member
    with the lowest guide rate (the same weighting `allocate_group_targets`
    already uses for allocation, applied here in reverse, as a triage
    order). `CON`/`PLUS_CON` are treated the same as `WELL`/`PLUG` here
    (the whole well is shut), since per-connection shut-in isn't enforced
    at the well level yet either.

    For `RATE` (`GECON`-only), cuts the group's own `target_rate` by
    `control_spec.group_rate_cutback_factor`.

    After every shut-in or rate cutback, reallocates the group's target
    across its remaining `GRUP`-mode members and re-resolves each one
    actually affected, before rechecking so a shut-in or cutback is never
    left half-applied. Repeats (shut another well, or cut again) until
    the group's limits are satisfied, no further eligible member well
    remains open, or `control_spec.max_fixed_point_iterations` outer
    passes are reached (best-effort, mirroring the well-level fixed-point
    solve's own iteration cap).

    For `NONE` (`GECON`'s own default), the limit is left violated and
    nothing is changed. It is just tracked, not enforced.

    :param group_name: Group to check. A row in `well_system.group_controls.names`.
    :param well_system: Supplies `.group_controls` (limits and
        membership) and `.controls`/`.well_kinds` (guide rates, read only).
    :param workspace: This run's `WellsWorkspace`, updated in place.
    :param control_spec: Supplies `group_rate_cutback_factor` and `max_fixed_point_iterations`.
    :param get_wellbore: Given a well row, its hydraulics correlation.
        Only called for a member actually being re-resolved.
    :param get_connection_samples: Given a well row, its active, open
        connections' current reservoir samples. Only called for a member
        actually being re-resolved.
    :param get_surface_fluid_properties: Given a well row, its surface
        fluid properties, or `None` if it has no THP control/limit to
        check. Omit if no affected member ever needs this.
    :returns: `GroupEconomicLimitOutcome` describing what was done.
    :raises ValidationError: If `well_system.group_controls` is `None`, or
        `group_name` isn't one of its rows.
    :raises StopSimulation: If a violated row's `end_run` breaches with no
        further eligible member well to shut, immediately after shutting
        one, or immediately for `RATE`/`NONE`.
    """
    group_controls = well_system.group_controls
    if group_controls is None:
        raise ValidationError("`well_system.group_controls` is not set.")

    try:
        group_row = group_controls.names.index(group_name)
    except ValueError:
        raise ValidationError(f"No `GroupControl` set for group {group_name!r}.") from None

    group_limits = group_controls.limits
    limits_start = group_limits.group_offsets[group_row]
    limits_end = group_limits.group_offsets[group_row + 1]
    if limits_start == limits_end:
        return GroupEconomicLimitOutcome(
            satisfied=True, shut_wells=(), rate_cutback_applied=False, reallocated_wells=()
        )

    get_resolved_surface_fluid_properties = get_surface_fluid_properties or (lambda well_row: None)

    member_start = group_controls.member_offsets[group_row]
    member_end = group_controls.member_offsets[group_row + 1]
    member_indices = group_controls.member_well_indices[member_start:member_end]

    expected_well_kind = (
        WellKind.INJECTOR
        if group_controls.group_kinds[group_row] == GroupKind.INJECTOR
        else WellKind.PRODUCER
    )

    controls = well_system.controls
    open_members = [
        i
        for i in member_indices
        if well_system.well_kinds[i] == expected_well_kind
        and workspace.economic_shutins[i] == 0
        and not math.isnan(workspace.bhps[i])
    ]

    grup_mode_tag = (
        InjectorControlModeTag.GROUP
        if expected_well_kind == WellKind.INJECTOR
        else ProducerControlModeTag.GROUP
    )
    grup_member_rows = tuple(i for i in open_members if controls.control_modes[i] == grup_mode_tag)

    shut_wells: list[str] = []
    reallocated_wells: list[str] = []
    rate_cutback_applied = False

    for _ in range(control_spec.max_fixed_point_iterations):
        phase_rates = aggregate_member_rates(workspace=workspace, open_member_wells=open_members)
        violated_row = check_economic_violation(
            limits=group_limits,
            limits_start=limits_start,
            limits_end=limits_end,
            phase_rates=phase_rates,
        )
        if violated_row == UNSET_INT:
            return GroupEconomicLimitOutcome(
                satisfied=True,
                shut_wells=tuple(shut_wells),
                rate_cutback_applied=rate_cutback_applied,
                reallocated_wells=tuple(reallocated_wells),
            )

        action = WorkoverActionTag(group_limits.workover_actions[violated_row])
        end_run = bool(group_limits.end_run_flags[violated_row])

        if action == WorkoverActionTag.NONE:
            return GroupEconomicLimitOutcome(
                satisfied=False,
                shut_wells=tuple(shut_wells),
                rate_cutback_applied=rate_cutback_applied,
                reallocated_wells=tuple(reallocated_wells),
            )

        if action == WorkoverActionTag.RATE:
            if end_run:
                raise StopSimulation(
                    f"Group {group_name!r} breached a `GECON` limit flagged to "
                    "end the run (workover action RATE)."
                )
            group_controls.target_rates[group_row] *= control_spec.group_rate_cutback_factor
            rate_cutback_applied = True
        else:
            # `WELL` / `PLUG` / `CON` / `PLUS_CON`: shut the eligible open member
            # with the lowest guide rate (NaN treated as the default
            # weight of 1.0, matching `allocate_group_targets`' own convention).
            if not open_members:
                if end_run:
                    raise StopSimulation(
                        f"Group {group_name!r} breached a `GECON` limit flagged to "
                        "end the run, with no eligible member well left to shut."
                    )
                return GroupEconomicLimitOutcome(
                    satisfied=False,
                    shut_wells=tuple(shut_wells),
                    rate_cutback_applied=rate_cutback_applied,
                    reallocated_wells=tuple(reallocated_wells),
                )

            guide_rates = controls.guide_rates[open_members]
            weights = np.where(np.isnan(guide_rates), 1.0, guide_rates)
            candidate = open_members[int(np.argmin(weights))]

            shut_in_member_well(well_system=well_system, workspace=workspace, well_row=candidate)
            shut_wells.append(well_system.names[candidate])
            open_members = [i for i in open_members if i != candidate]

            if end_run:
                raise StopSimulation(
                    f"Group {group_name!r} breached a `GECON` limit flagged to end "
                    f"the run; shut in well {well_system.names[candidate]!r}."
                )

        reresolved = reallocate_and_reresolve(
            group_name=group_name,
            well_system=well_system,
            workspace=workspace,
            control_spec=control_spec,
            open_members=open_members,
            grup_mode_tag=grup_mode_tag,
            grup_member_rows=grup_member_rows,
            get_wellbore=get_wellbore,
            get_connection_samples=get_connection_samples,
            get_surface_fluid_properties=get_resolved_surface_fluid_properties,
        )
        reallocated_wells.extend(name for name in reresolved if name not in reallocated_wells)

    phase_rates = aggregate_member_rates(workspace=workspace, open_member_wells=open_members)
    still_violated = (
        check_economic_violation(
            limits=group_limits,
            limits_start=limits_start,
            limits_end=limits_end,
            phase_rates=phase_rates,
        )
        != UNSET_INT
    )
    return GroupEconomicLimitOutcome(
        satisfied=not still_violated,
        shut_wells=tuple(shut_wells),
        rate_cutback_applied=rate_cutback_applied,
        reallocated_wells=tuple(reallocated_wells),
    )
