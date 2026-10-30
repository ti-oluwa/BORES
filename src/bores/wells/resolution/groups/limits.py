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

import numba
import numpy as np

from bores.errors import StopSimulation, ValidationError
from bores.types import Boolean, IntArray, Integer, NumberArray, OneDimension, PhaseValues
from bores.wells.compile import (
    UNSET_INT,
    CompiledWellSystem,
    GroupKind,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
    WorkoverActionTag,
)
from bores.wells.hydraulics.base import ConnectionFluids, SurfaceSample, WellBoreModel
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.resolution.engine import resolve_well_control
from bores.wells.resolution.groups.allocation import allocate_group_targets
from bores.wells.resolution.limits import check_economic_violation
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.workspace import WellsWorkspace

__all__ = ["GroupEconomicLimitOutcome", "enforce_group_economic_limits"]


class GroupEconomicLimitOutcome(typing.NamedTuple):
    """Result of one `enforce_group_economic_limits` call."""

    satisfied: Boolean
    """Whether every one of this group's `GECON` limits is satisfied when this call returns."""

    shut_wells: tuple[str, ...]
    """Names of member wells shut this call, in the order they were shut."""

    rate_cutback_applied: Boolean
    """Whether this call cut the group's own `target_rate` (`workover_action` was `RATE`)."""

    reallocated_wells: tuple[str, ...]
    """
    Names of `GRUP`-mode member wells whose target was recomputed and who
    were actually re-resolved this call, following a shut-in or a rate cutback.
    """


@numba.njit(cache=True)
def sum_member_rates(
    *,
    surface_oil_rates: NumberArray[OneDimension],
    surface_water_rates: NumberArray[OneDimension],
    surface_gas_rates: NumberArray[OneDimension],
    members: IntArray[OneDimension],
) -> PhaseValues:
    """
    Sums the surface-condition rates of the given wells.

    :param surface_oil_rates: Every well's resolved surface oil rate, indexed by well row.
    :param surface_water_rates: Every well's resolved surface water rate, indexed by well row.
    :param surface_gas_rates: Every well's resolved surface gas rate, indexed by well row.
    :param members: Well rows to sum over.
    :returns: The summed `PhaseValues`.
    """
    oil = 0.0
    water = 0.0
    gas = 0.0
    for position in range(members.shape[0]):
        well_row = members[position]
        oil += surface_oil_rates[well_row]
        water += surface_water_rates[well_row]
        gas += surface_gas_rates[well_row]
    return PhaseValues(oil=oil, water=water, gas=gas)  # type: ignore[arg-type]


def aggregate_member_rates(
    workspace: WellsWorkspace, open_member_wells: typing.Sequence[Integer] | IntArray[OneDimension]
) -> PhaseValues:
    """
    Sums already-resolved surface-condition rates over `open_member_wells`, which is what
    economic limits are written against. The gas rate is the total, free and dissolved.

    :param workspace: Supplies each member well's own resolved rates.
    :param open_member_wells: Rows to sum over.
    :returns: The group's current aggregate `PhaseValues`.
    """
    return sum_member_rates(
        surface_oil_rates=workspace.surface_oil_rates,
        surface_water_rates=workspace.surface_water_rates,
        surface_gas_rates=workspace.surface_gas_rates,
        members=typing.cast(
            "IntArray[OneDimension]", np.asarray(open_member_wells, dtype=np.int64)
        ),
    )


@numba.njit(cache=True)
def select_open_members(
    *,
    member_indices: IntArray[OneDimension],
    well_kinds: IntArray[OneDimension],
    expected_well_kind: Integer,
    economic_shutins: IntArray[OneDimension],
    bhps: NumberArray[OneDimension],
) -> IntArray[OneDimension]:
    """
    Picks the group members that are still open: of the group's own well kind, not shut in
    economically, and resolved this pass (a resolved well has a bottom-hole pressure).

    :param member_indices: Well rows of the group's members.
    :param well_kinds: Every well's kind tag, indexed by well row.
    :param expected_well_kind: The `WellKind` the group's members must be.
    :param economic_shutins: Every well's economic shut-in flag, indexed by well row.
    :param bhps: Every well's resolved bottom-hole pressure, indexed by well row.
    :returns: The open members' well rows, in the order of `member_indices`.
    """
    out = np.empty(member_indices.shape[0], dtype=np.int64)
    count = 0
    for position in range(member_indices.shape[0]):
        well_row = member_indices[position]
        if (
            well_kinds[well_row] == expected_well_kind
            and economic_shutins[well_row] == 0
            and not math.isnan(bhps[well_row])
        ):
            out[count] = well_row
            count += 1
    return out[:count]  # type: ignore[return-value]


@numba.njit(cache=True)
def select_members_in_mode(
    *,
    members: IntArray[OneDimension],
    control_modes: IntArray[OneDimension],
    mode_tag: Integer,
) -> IntArray[OneDimension]:
    """
    :param members: Well rows to filter.
    :param control_modes: Every well's control mode tag, indexed by well row.
    :param mode_tag: The control mode tag to keep.
    :returns: The rows of `members` currently in `mode_tag`, in order.
    """
    out = np.empty(members.shape[0], dtype=np.int64)
    count = 0
    for position in range(members.shape[0]):
        if control_modes[members[position]] == mode_tag:
            out[count] = members[position]
            count += 1
    return out[:count]  # type: ignore[return-value]


@numba.njit(cache=True)
def find_lowest_guide_rate_member(
    *, members: IntArray[OneDimension], guide_rates: NumberArray[OneDimension]
) -> Integer:
    """
    Finds the member with the lowest guide rate, the first to shut in when a group limit is
    breached. A `NaN` guide rate counts as 1, the same default weight group allocation uses. The
    first of equally weighted members wins.

    :param members: Well rows to choose from. Must not be empty.
    :param guide_rates: Every well's guide rate, indexed by well row.
    :returns: The chosen member's position in `members`.
    """
    best_position = 0
    best_weight = guide_rates[members[0]]
    if math.isnan(best_weight):
        best_weight = 1.0
    for position in range(1, members.shape[0]):
        weight = guide_rates[members[position]]
        if math.isnan(weight):
            weight = 1.0
        if weight < best_weight:
            best_weight = weight
            best_position = position
    return best_position


def shut_in_member_well(
    well_system: CompiledWellSystem, workspace: WellsWorkspace, well_row: Integer
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
    spec: WellControlSpec,
    open_members: typing.Sequence[Integer],
    grup_mode_tag: Integer,
    grup_member_rows: typing.Sequence[Integer],
    get_tubing_performance: typing.Callable[[Integer], WellBoreModel | VFPTables | None],
    get_connection_fluids: typing.Callable[[Integer], ConnectionFluids],
    get_surface_sample: typing.Callable[[Integer], SurfaceSample | None],
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
    :param get_tubing_performance: Given a well row, what relates its BHP to its THP
        (a `WellBoreModel` or a `VFPTables` collection), or `None` if it has none.
    :param get_connection_fluids: Given a well row, its active, open
        connections' current reservoir conditions.
    :param get_surface_sample: Given a well row, its surface
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
            tubing_performance=get_tubing_performance(well_row),
            connection_fluids=get_connection_fluids(well_row),
            workspace=workspace,
            spec=spec,
            surface_sample=get_surface_sample(well_row),
        )
    return tuple(reresolved)


def enforce_group_economic_limits(
    *,
    group_name: str,
    well_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    spec: WellControlSpec,
    get_tubing_performance: typing.Callable[[Integer], WellBoreModel | VFPTables | None],
    get_connection_fluids: typing.Callable[[Integer], ConnectionFluids],
    get_surface_sample: typing.Callable[[Integer], SurfaceSample | None] | None = None,
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
    `spec.group_rate_cutback_factor`.

    After every shut-in or rate cutback, reallocates the group's target
    across its remaining `GRUP`-mode members and re-resolves each one
    actually affected, before rechecking so a shut-in or cutback is never
    left half-applied. Repeats (shut another well, or cut again) until
    the group's limits are satisfied, no further eligible member well
    remains open, or `spec.max_fixed_point_iterations` outer
    passes are reached (best-effort, mirroring the well-level fixed-point
    solve's own iteration cap).

    For `NONE` (`GECON`'s own default), the limit is left violated and
    nothing is changed. It is just tracked, not enforced.

    :param group_name: Group to check. A row in `well_system.group_controls.names`.
    :param well_system: Supplies `.group_controls` (limits and
        membership) and `.controls`/`.well_kinds` (guide rates, read only).
    :param workspace: This run's `WellsWorkspace`, updated in place.
    :param spec: Supplies `group_rate_cutback_factor` and `max_fixed_point_iterations`.
    :param get_tubing_performance: Given a well row, what relates its BHP to its THP
        (a `WellBoreModel` or a `VFPTables` collection), or `None` if it has none. Only called for a member actually being
        re-resolved.
    :param get_connection_fluids: Given a well row, its active, open
        connections' current reservoir conditions. Only called for a member
        actually being re-resolved.
    :param get_surface_sample: Given a well row, its surface
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

    group_row = group_controls.group_row(name=group_name)

    group_limits = group_controls.limits
    limits_start = group_limits.group_offsets[group_row]
    limits_end = group_limits.group_offsets[group_row + 1]
    if limits_start == limits_end:
        return GroupEconomicLimitOutcome(
            satisfied=True, shut_wells=(), rate_cutback_applied=False, reallocated_wells=()
        )

    get_resolved_surface_sample = get_surface_sample or (lambda well_row: None)

    member_start = group_controls.member_offsets[group_row]
    member_end = group_controls.member_offsets[group_row + 1]
    member_indices = group_controls.member_well_indices[member_start:member_end]

    expected_well_kind = (
        WellKind.INJECTOR
        if group_controls.group_kinds[group_row] == GroupKind.INJECTOR
        else WellKind.PRODUCER
    )

    controls = well_system.controls
    open_members = select_open_members(
        member_indices=typing.cast("IntArray[OneDimension]", member_indices),
        well_kinds=well_system.well_kinds,
        expected_well_kind=expected_well_kind,
        economic_shutins=workspace.economic_shutins,
        bhps=workspace.bhps,
    )

    grup_mode_tag = (
        InjectorControlModeTag.GROUP
        if expected_well_kind == WellKind.INJECTOR
        else ProducerControlModeTag.GROUP
    )
    grup_member_rows = select_members_in_mode(
        members=open_members,
        control_modes=controls.control_modes,
        mode_tag=grup_mode_tag,
    )

    shut_wells: list[str] = []
    reallocated_wells: list[str] = []
    rate_cutback_applied = False

    for _ in range(spec.max_fixed_point_iterations):
        rates = aggregate_member_rates(
            workspace=workspace,
            open_member_wells=typing.cast("IntArray[OneDimension]", open_members),
        )
        violated_row = check_economic_violation(
            limits=group_limits,
            limits_start=limits_start,
            limits_end=limits_end,
            rates=rates,
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
            group_controls.target_rates[group_row] *= spec.group_rate_cutback_factor
            rate_cutback_applied = True
        else:
            # `WELL` / `PLUG` / `CON` / `PLUS_CON`: shut the eligible open member
            # with the lowest guide rate (NaN treated as the default
            # weight of 1.0, matching `allocate_group_targets`' own convention).
            if open_members.shape[0] == 0:
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

            candidate = int(
                open_members[
                    find_lowest_guide_rate_member(
                        members=typing.cast("IntArray[OneDimension]", open_members),
                        guide_rates=controls.guide_rates,
                    )
                ]
            )

            shut_in_member_well(well_system=well_system, workspace=workspace, well_row=candidate)
            shut_wells.append(well_system.names[candidate])
            open_members = open_members[open_members != candidate]

            if end_run:
                raise StopSimulation(
                    f"Group {group_name!r} breached a `GECON` limit flagged to end "
                    f"the run; shut in well {well_system.names[candidate]!r}."
                )

        reresolved = reallocate_and_reresolve(
            group_name=group_name,
            well_system=well_system,
            workspace=workspace,
            spec=spec,
            open_members=typing.cast("typing.Sequence[Integer]", open_members),
            grup_mode_tag=grup_mode_tag,
            grup_member_rows=typing.cast("typing.Sequence[Integer]", grup_member_rows),
            get_tubing_performance=get_tubing_performance,
            get_connection_fluids=get_connection_fluids,
            get_surface_sample=get_resolved_surface_sample,
        )
        reallocated_wells.extend(name for name in reresolved if name not in reallocated_wells)

    rates = aggregate_member_rates(
        workspace=workspace, open_member_wells=typing.cast("IntArray[OneDimension]", open_members)
    )
    still_violated = (
        check_economic_violation(
            limits=group_limits,
            limits_start=limits_start,
            limits_end=limits_end,
            rates=rates,
        )
        != UNSET_INT
    )
    return GroupEconomicLimitOutcome(
        satisfied=not still_violated,
        shut_wells=tuple(shut_wells),
        rate_cutback_applied=rate_cutback_applied,
        reallocated_wells=tuple(reallocated_wells),
    )
