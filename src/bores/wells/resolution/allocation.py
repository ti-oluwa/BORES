"""
`CompiledGroupControls` target allocation. Splits a group's target rate
across its eligible member wells by guide rate, writing concrete per-well
`target_rates`/`control_modes` directly into `CompiledWellControls` in
place for wells whose mode is `GRUP`.

First-pass proportional allocation only: does not iterate to redistribute
a shortfall when an allocated well can't actually reach its share (e.g.
hits its own BHP limit). That redistribution loop is not implemented.
"""

import numpy as np

from bores.errors import ValidationError
from bores.wells.compile import (
    GROUP_TO_INJECTOR_MODE_TAG,
    GROUP_TO_PRODUCER_MODE_TAG,
    CompiledWellSystem,
    GroupKind,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
)
from bores.wells.workspace import WellsWorkspace

__all__ = ["allocate_group_targets"]


def allocate_group_targets(
    group_name: str, well_system: CompiledWellSystem, workspace: WellsWorkspace
) -> tuple[str, ...]:
    """
    Allocate `group_name`'s current target rate across its member wells
    whose control mode is `GRUP`, by guide rate.

    Writes the resulting per-well control (mode switched to a concrete
    rate mode, `target_rates` set to the well's share) directly into
    `well_system.controls.target_rates`/`.control_modes` in place.

    Member wells were resolved once at compile time (`compile_group_controls`);
    this only re-evaluates which of them currently sit in `GRUP` mode,
    since that's dynamic. A member currently shut in (`workspace.economic_shutins`)
    is excluded from both the weighting and the allocation, whatever its
    own `control_mode` still says, since it isn't actually available to
    take a share of the group's target right now.

    :param group_name: Group to allocate. A row in  `well_system.group_controls.names`.
    :param well_system: Supplies `.group_controls` (target and compiled
        membership) and `.controls`/`.well_kinds` (written to in place).
    :param workspace: This run's `WellsWorkspace`. Read only, for each
        member's current `economic_shutins` flag.
    :returns: Names of the wells actually allocated, for the eligible
        member wells (empty if none are eligible).
    :raises ValidationError: If `well_system.group_controls` is `None`, or
        `group_name` isn't one of its rows, or its `mode` has no
        allocatable target (`FLD`/`NONE`/`VREP`/`REIN`).
    """
    group_controls = well_system.group_controls
    if group_controls is None:
        raise ValidationError("`well_system.group_controls` is not set.")

    try:
        group_index = group_controls.names.index(group_name)
    except ValueError:
        raise ValidationError(f"No `GroupControl` set for group {group_name!r}.") from None

    group_kind = group_controls.group_kinds[group_index]
    group_mode = group_controls.control_modes[group_index]
    is_injection = group_kind == GroupKind.INJECTOR
    target_mode_tag = (
        GROUP_TO_INJECTOR_MODE_TAG.get(group_mode)
        if is_injection
        else GROUP_TO_PRODUCER_MODE_TAG.get(group_mode)
    )
    if target_mode_tag is None:
        raise ValidationError(
            f"Group {group_name!r}'s control mode has no directly allocatable rate target."
        )

    target_rate = group_controls.target_rates[group_index]
    if np.isnan(target_rate):
        raise ValidationError(f"Group {group_name!r}'s control has no target_rate.")

    member_start = group_controls.member_offsets[group_index]
    member_end = group_controls.member_offsets[group_index + 1]
    member_indices = group_controls.member_well_indices[member_start:member_end]

    grup_mode_tag = InjectorControlModeTag.GROUP if is_injection else ProducerControlModeTag.GROUP
    expected_well_kind = WellKind.INJECTOR if is_injection else WellKind.PRODUCER
    controls = well_system.controls
    eligible = [
        i
        for i in member_indices
        if well_system.well_kinds[i] == expected_well_kind
        and controls.control_modes[i] == grup_mode_tag
        and workspace.economic_shutins[i] == 0
    ]
    if not eligible:
        return ()

    guide_rates = controls.guide_rates[eligible]
    weights = np.where(np.isnan(guide_rates), 1.0, guide_rates)
    total_weight = weights.sum()

    for i, weight in zip(eligible, weights, strict=False):
        controls.target_rates[i] = target_rate * (weight / total_weight)
        controls.control_modes[i] = target_mode_tag

    return tuple(well_system.names[i] for i in eligible)
