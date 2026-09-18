"""
`GECON` group-level economic-limit enforcement.

Checked after a group's member wells have already been resolved for the
current timestep (and, for a `GRUP`-mode member, after
`wells.resolution.allocation.allocate_group_targets` has given it a
concrete target). Aggregates already-resolved member rates rather than
resolving anything itself.
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
    WellKind,
    WorkoverActionTag,
)
from bores.wells.resolution.limits import check_economic_violation
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.states import PhaseValues
from bores.wells.workspace import WellsWorkspace

__all__ = ["GroupEconomicLimitOutcome", "enforce_group_economic_limits"]


class GroupEconomicLimitOutcome(typing.NamedTuple):
    """Result of one `enforce_group_economic_limits` call."""

    violated_row: Integer
    """
    The last-checked violated row in this group's `CompiledGroupLimits`,
    or `UNSET_INT` if the group satisfies every limit (including after
    every action this call could take).
    """

    workover_action: WorkoverActionTag | None
    """`violated_row`'s own action, or `None` if `violated_row` is `UNSET_INT`."""

    shut_wells: tuple[str, ...]
    """
    Names of member wells shut this call, in the order they were shut.
    Empty unless `workover_action` was `WELL`/`PLUG`/`CON`/`PLUS_CON`.
    """

    rate_cutback_applied: bool
    """
    Whether this call cut the group's own `target_rate`
    (`workover_action` was `RATE`). The caller still needs to
    re-run `allocate_group_targets` and re-resolve affected member wells
    for the cutback to take effect, and call this again afterward to
    check whether it was enough.
    """


def _aggregate_member_rates(
    *,
    workspace: WellsWorkspace,
    open_member_wells: typing.Sequence[Integer],
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


def _shut_in_member_well(
    *,
    well_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    well_row: Integer,
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


def enforce_group_economic_limits(
    *,
    group_name: str,
    well_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
) -> GroupEconomicLimitOutcome:
    """
    Checks `group_name`'s own `GECON` limits against its member wells'
    already-resolved rates, and acts on the first one violated.

    For `WELL`/`PLUG`/`CON`/`PLUS_CON`, shuts in the eligible open member
    well with the lowest guide rate (the same weighting
    `allocate_group_targets` already uses for allocation, applied here in
    reverse, as a triage order), rechecking the group's aggregate after
    each shut-in, until the limit is satisfied or no eligible member well
    remains open. `CON`/`PLUS_CON` are treated the same as `WELL`/`PLUG`
    here (the whole well is shut), since per-connection shut-in isn't
    enforced at the well level yet either.

    For `RATE` (`GECON`-only), cuts the group's own `target_rate` by
    `control_spec.group_rate_cutback_factor` and stops, rather than
    shutting any member well - satisfying a rate cutback needs the
    group's targets reallocated and the affected members re-resolved,
    both outside this function's own scope. Call `allocate_group_targets`
    and re-resolve, then call this again to check whether it was enough.

    For `NONE` (`GECON`'s own default), the limit is left violated and
    nothing is changed - tracked, not enforced.

    :param group_name: Group to check. A row in `well_system.group_controls.names`.
    :param well_system: Supplies `.group_controls` (limits and
        membership) and `.controls`/`.well_kinds` (guide rates, read only).
    :param workspace: This run's `WellsWorkspace`, updated in place for
        any well actually shut in.
    :param control_spec: Supplies `group_rate_cutback_factor`.
    :returns: `GroupEconomicLimitOutcome` describing what, if anything, was done.
    :raises ValidationError: If `well_system.group_controls` is `None`, or
        `group_name` isn't one of its rows.
    :raises StopSimulation: If a violated row's `end_run` breaches with no
        further eligible member well to shut (`WELL`/`PLUG`/`CON`/`PLUS_CON`),
        immediately after shutting one, or immediately for `RATE`/`NONE`.
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
            violated_row=UNSET_INT, workover_action=None, shut_wells=(), rate_cutback_applied=False
        )

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

    shut_wells: list[str] = []
    violated_row = UNSET_INT
    action: WorkoverActionTag | None = None
    rate_cutback_applied = False

    while True:
        phase_rates = _aggregate_member_rates(
            workspace=workspace, open_member_wells=open_members
        )
        violated_row = check_economic_violation(
            limits=group_limits,
            limits_start=limits_start,
            limits_end=limits_end,
            phase_rates=phase_rates,
        )
        if violated_row == UNSET_INT:
            action = None
            break

        action = WorkoverActionTag(group_limits.workover_actions[violated_row])
        end_run = bool(group_limits.end_run_flags[violated_row])

        if action == WorkoverActionTag.NONE:
            break

        if action == WorkoverActionTag.RATE:
            if end_run:
                raise StopSimulation(
                    f"Group {group_name!r} breached a GECON limit flagged to "
                    "end the run (workover action RATE)."
                )
            group_controls.target_rates[group_row] *= control_spec.group_rate_cutback_factor
            rate_cutback_applied = True
            break

        # WELL / PLUG / CON / PLUS_CON: shut the eligible open member with
        # the lowest guide rate (NaN treated as the default weight of 1.0,
        # matching allocate_group_targets' own convention).
        if not open_members:
            if end_run:
                raise StopSimulation(
                    f"Group {group_name!r} breached a GECON limit flagged to "
                    "end the run, with no eligible member well left to shut."
                )
            break

        guide_rates = controls.guide_rates[open_members]
        weights = np.where(np.isnan(guide_rates), 1.0, guide_rates)
        candidate = open_members[int(np.argmin(weights))]

        _shut_in_member_well(well_system=well_system, workspace=workspace, well_row=candidate)
        shut_wells.append(well_system.names[candidate])
        open_members = [i for i in open_members if i != candidate]

        if end_run:
            raise StopSimulation(
                f"Group {group_name!r} breached a GECON limit flagged to end "
                f"the run; shut in well {well_system.names[candidate]!r}."
            )

    return GroupEconomicLimitOutcome(
        violated_row=violated_row,
        workover_action=action,
        shut_wells=tuple(shut_wells),
        rate_cutback_applied=rate_cutback_applied,
    )
