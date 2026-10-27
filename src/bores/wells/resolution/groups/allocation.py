"""
`CompiledGroupControls` target allocation. Splits a group's target rate
across its eligible_member_indices member wells by guide rate, writing concrete per-well
`target_rates`/`control_modes` directly into `CompiledWellControls` in
place for wells whose mode is `GRUP`.

`allocate_group_targets` itself is a single first-pass proportional
split, with no resolve dependency, usable anywhere a concrete per-well
target is needed without also resolving anything. `redistribute_group_shortfall`
builds on it: after an initial resolve, checks whether every allocated
member actually reached its own share, and if not, grows the target of
whichever members have headroom and re-resolves them, repeating until
the group's real target is met or no member has any headroom left.
"""

import math
import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.types import Integer, Number
from bores.wells.compile import (
    GROUP_TO_INJECTOR_MODE_TAG,
    GROUP_TO_PRODUCER_MODE_TAG,
    CompiledWellSystem,
    FluidPhaseTag,
    GroupKind,
    InjectorControlModeTag,
    ProducerControlModeTag,
    WellKind,
)
from bores.wells.hydraulics.base import ConnectionFluids, SurfaceSample, WellBoreModel
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.resolution.engine import resolve_well_control
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.workspace import WellsWorkspace

__all__ = ["allocate_group_targets", "get_group_controlled_rate", "redistribute_group_shortfall"]


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
    take a share of the group's target right now. The same goes for a member
    that cannot flow: one not yet in effect, one that is shut, or one with no
    open connection in effect.

    :param group_name: Group to allocate. A row in  `well_system.group_controls.names`.
    :param well_system: Supplies `.group_controls` (target and compiled
        membership) and `.controls`/`.well_kinds` (written to in place).
    :param workspace: This run's `WellsWorkspace`. Read only, for each
        member's current `economic_shutins` flag.
    :returns: Names of the wells actually allocated, for the eligible_member_indices
        member wells (empty if none are eligible_member_indices).
    :raises ValidationError: If `well_system.group_controls` is `None`, or
        `group_name` isn't one of its rows, or its `mode` has no
        allocatable target (`FLD`/`NONE`/`VREP`/`REIN`).
    """
    group_controls = well_system.group_controls
    if group_controls is None:
        raise ValidationError("`well_system.group_controls` is not set.")

    group_row = group_controls.group_row(name=group_name)

    # TODO: Possible application of njit from here downward (can be section or parts,
    # njitted separately of the whole thing). Just like in `redistribute_group_shortfall`
    group_kind = group_controls.group_kinds[group_row]
    group_mode = group_controls.control_modes[group_row]
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

    target_rate = group_controls.target_rates[group_row]
    if np.isnan(target_rate):
        raise ValidationError(f"Group {group_name!r}'s control has no `target_rate`.")

    member_start = group_controls.member_offsets[group_row]
    member_end = group_controls.member_offsets[group_row + 1]
    member_indices = group_controls.member_well_indices[member_start:member_end]

    group_mode_tag = InjectorControlModeTag.GROUP if is_injection else ProducerControlModeTag.GROUP
    expected_well_kind = WellKind.INJECTOR if is_injection else WellKind.PRODUCER
    controls = well_system.controls
    eligible_member_indices = [
        idx
        for idx in member_indices
        if well_system.well_kinds[idx] == expected_well_kind
        and controls.control_modes[idx] == group_mode_tag
        and workspace.economic_shutins[idx] == 0
        and len(well_system.get_flowing_connection_rows(well_row=idx)) > 0
    ]
    if not eligible_member_indices:
        return ()

    guide_rates = controls.guide_rates[eligible_member_indices]
    weights = np.where(np.isnan(guide_rates), 1.0, guide_rates)
    total_weight = weights.sum()

    for idx, weight in zip(eligible_member_indices, weights, strict=False):
        controls.target_rates[idx] = target_rate * (weight / total_weight)
        controls.control_modes[idx] = target_mode_tag

    return tuple(well_system.names[idx] for idx in eligible_member_indices)


@numba.njit(cache=True)
def get_group_controlled_rate(
    *, well_system: CompiledWellSystem, workspace: WellsWorkspace, well_row: Integer
) -> Number:
    """
    A member well's own resolved rate, in whatever quantity and
    condition (surface or reservoir) its current concrete control mode
    actually targets. This is the one value directly comparable against
    `CompiledWellControls.target_rates[well_row]` once
    `allocate_group_targets` has converted it from `GRUP`.

    `LIQUID_RATE` is `surface_oil_rates + surface_water_rates`.
    `RESERVOIR_VOLUME_RATE` is the reservoir-condition total across all
    three phases. Both are plain sums of already-resolved rates, not a
    separately tracked quantity.

    :param well_system: Supplies `.controls`/`.well_kinds`.
    :param workspace: Supplies each well's own resolved rates.
    :param well_row: The well to read. Must already be resolved.
    :returns: The resolved rate, in the same quantity and condition as
        this well's own `target_rates` entry.
    :raises ValidationError: If this well's current control mode has no
        rate target (`BHP`/`THP`/`GROUP`/`UNSET`).
    """
    controls = well_system.controls
    mode = controls.control_modes[well_row]

    if well_system.well_kinds[well_row] == WellKind.INJECTOR:
        if mode == InjectorControlModeTag.RATE:
            injected_phase = controls.injected_phases[well_row]
            if injected_phase == FluidPhaseTag.OIL:
                return workspace.surface_oil_rates[well_row]
            elif injected_phase == FluidPhaseTag.WATER:
                return workspace.surface_water_rates[well_row]
            return workspace.surface_gas_rates[well_row]
        if mode == InjectorControlModeTag.RESERVOIR_VOLUME_RATE:
            return (
                workspace.oil_rates[well_row]
                + workspace.water_rates[well_row]
                + workspace.gas_rates[well_row]
            )
        raise ValidationError(
            f"Well row {well_row} has no rate target under its current control mode."
        )

    if mode == ProducerControlModeTag.OIL_RATE:
        return workspace.surface_oil_rates[well_row]
    if mode == ProducerControlModeTag.WATER_RATE:
        return workspace.surface_water_rates[well_row]
    if mode == ProducerControlModeTag.GAS_RATE:
        return workspace.surface_gas_rates[well_row]
    if mode == ProducerControlModeTag.LIQUID_RATE:
        return workspace.surface_oil_rates[well_row] + workspace.surface_water_rates[well_row]
    if mode == ProducerControlModeTag.RESERVOIR_VOLUME_RATE:
        return (
            workspace.oil_rates[well_row]
            + workspace.water_rates[well_row]
            + workspace.gas_rates[well_row]
        )
    raise ValidationError(
        f"Well row {well_row} has no rate target under its current control mode."
    )


@numba.njit(cache=True)
def cap_unmet_targets(
    *,
    target_rates: npt.NDArray,
    actual_rates: npt.NDArray,
    members: npt.NDArray,
    rate_tolerance: Number,
    headroom_members: npt.NDArray,
) -> tuple[Number, Integer]:
    """
    Finds which group members fell short of their own target rate and which did not.

    A member that delivered less than its target, by more than the tolerance, is bound below it
    (its BHP floor, say). Its target is lowered to what it delivered, so the same gap is not counted
    again on the next pass, and the gap is added to the group's shortfall. Every other member with
    a target has headroom. A member with no target (`NaN`) is skipped.

    :param target_rates: Every well's target rate, indexed by well row. Lowered in place for
        members that fell short.
    :param actual_rates: The rate each member delivered, in the order of `members`. Only read for
        a member with a target.
    :param members: Well rows of the group's allocated members.
    :param rate_tolerance: Relative tolerance on a member reaching its target. Used as an absolute
        tolerance for a target of exactly zero.
    :param headroom_members: Buffer the well rows of members with headroom are written to, from
        the start. Must hold at least `len(members)` entries.
    :returns: The group's total shortfall, and how many leading entries of `headroom_members`
        were written.
    """
    total_shortfall = 0.0
    n_headroom = 0
    for position in range(len(members)):
        well_row = members[position]
        target_rate = target_rates[well_row]
        if math.isnan(target_rate):
            continue
        actual_rate = actual_rates[position]
        gap = target_rate - actual_rate
        tolerance = abs(target_rate) * rate_tolerance if target_rate != 0 else rate_tolerance
        if gap > tolerance:
            total_shortfall += gap
            target_rates[well_row] = actual_rate
        elif gap <= tolerance:
            headroom_members[n_headroom] = well_row
            n_headroom += 1
    return total_shortfall, n_headroom


@numba.njit(cache=True)
def boost_headroom_targets(
    *,
    target_rates: npt.NDArray,
    guide_rates: npt.NDArray,
    headroom_members: npt.NDArray,
    n_headroom: Integer,
    total_shortfall: Number,
) -> None:
    """
    Adds a group's shortfall to the targets of the members with headroom, by guide rate.

    :param target_rates: Every well's target rate, indexed by well row. Raised in place.
    :param guide_rates: Every well's guide rate, indexed by well row. A `NaN` guide rate counts
        as 1.
    :param headroom_members: Well rows of the members with headroom, from the start.
    :param n_headroom: How many leading entries of `headroom_members` to use.
    :param total_shortfall: The shortfall to share out.
    """
    total_weight = 0.0
    for position in range(n_headroom):
        weight = guide_rates[headroom_members[position]]
        total_weight += 1.0 if math.isnan(weight) else weight
    for position in range(n_headroom):
        well_row = headroom_members[position]
        weight = guide_rates[well_row]
        weight = 1.0 if math.isnan(weight) else weight
        target_rates[well_row] += total_shortfall * (weight / total_weight)


def redistribute_group_shortfall(
    *,
    group_name: str,
    allocated_wells: typing.Sequence[str],
    well_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
    get_tubing_performance: typing.Callable[[Integer], WellBoreModel | VFPTables | None],
    get_connection_fluids: typing.Callable[[Integer], ConnectionFluids],
    get_surface_sample: typing.Callable[[Integer], SurfaceSample | None] | None = None,
) -> tuple[str, ...]:
    """
    Redistributes a group's shortfall across members with headroom.

    `allocate_group_targets` only does a first-pass proportional split.
    A member that can't actually reach its own share (its own BHP floor
    binds it first, say) leaves the group short of its real target, with
    nothing correcting for it. This looks for exactly that; a bound
    member has its own target capped down to what it actually delivered
    (so the same gap isn't counted again next pass), and its unmet share
    is added to this pass's total shortfall. Any member already sitting
    at or above its own target has headroom (nothing bound it below what
    it was asked for), so its target grows by its guide-rate share of
    that shortfall, and it's re-resolved. Repeats, rechecking every
    member each pass, until the shortfall clears, no member has headroom
    left, or `control_spec.max_fixed_point_iterations` passes are used.

    Requires `allocate_group_targets` and an initial resolve of every
    member in `allocated_wells` to have already run this timestep. Reads
    their current resolved rates via `get_group_controlled_rate`, it
    doesn't resolve anything itself before its own first pass.

    :param group_name: Group to redistribute.
    :param allocated_wells: Exactly the names `allocate_group_targets`
        returned for this group this pass. Deliberately not re-derived
        here by inspecting each member's current control mode, since a
        well on `BHP`/`THP` control for a reason unrelated to this
        group's own allocation would otherwise be wrongly swept in.
    :param well_system: Supplies `.group_controls`/`.controls`/`.well_kinds`.
    :param workspace: This run's `WellsWorkspace`, updated in place.
    :param control_spec: Supplies `max_fixed_point_iterations` and `rate_convergence_tolerance`.
    :param get_tubing_performance: Given a well row, what relates its BHP to its THP
        (a `WellBoreModel` or a `VFPTables` collection), or `None` if it has none. Only called
        for a member actually being re-resolved.
    :param get_connection_fluids: Given a well row, its active, open
        connections' current reservoir conditions. Only called for a member
        actually being re-resolved.
    :param get_surface_sample: Given a well row, its surface
        fluid properties, or `None` if it has no THP control/limit to
        check. Omit if no affected member ever needs this.
    :returns: Names of the members actually re-resolved with a boosted
        target, in the order they were last touched (a member re-resolved
        more than once appears only at its most recent position).
    :raises ValidationError: If `well_system.group_controls` is `None`,
        `group_name` isn't one of its rows, or a member has a control
        mode `get_group_controlled_rate` can't read a rate from.
    """
    group_controls = well_system.group_controls
    if group_controls is None:
        raise ValidationError("`well_system.group_controls` is not set.")

    # Resolve `group_name` up front, purely to validate it.
    group_controls.group_row(name=group_name)

    if not allocated_wells:
        return ()

    controls = well_system.controls
    group_derived = [well_system.well_row(name=name) for name in allocated_wells]

    resolved_get_surface_sample = get_surface_sample or (lambda well_row: None)
    reresolved: list[str] = []

    members = np.asarray(group_derived, dtype=np.int64)
    actual_rates = np.empty(len(members), dtype=np.float64)
    headroom_members = np.empty(len(members), dtype=np.int64)
    for _ in range(control_spec.max_fixed_point_iterations):
        for position, idx in enumerate(group_derived):
            actual_rates[position] = (
                math.nan
                if math.isnan(controls.target_rates[idx])
                else get_group_controlled_rate(
                    well_system=well_system, workspace=workspace, well_row=idx
                )
            )

        total_shortfall, n_headroom = cap_unmet_targets(
            target_rates=controls.target_rates,
            actual_rates=actual_rates,
            members=members,
            rate_tolerance=control_spec.rate_convergence_tolerance,
            headroom_members=headroom_members,
        )
        if total_shortfall <= 0 or n_headroom == 0:
            break

        boost_headroom_targets(
            target_rates=controls.target_rates,
            guide_rates=controls.guide_rates,
            headroom_members=headroom_members,
            n_headroom=n_headroom,
            total_shortfall=total_shortfall,
        )
        for idx in headroom_members[:n_headroom]:
            resolve_well_control(
                compiled_system=well_system,
                well_row=idx,
                tubing_performance=get_tubing_performance(idx),
                connection_fluids=get_connection_fluids(idx),
                workspace=workspace,
                control_spec=control_spec,
                surface_sample=resolved_get_surface_sample(idx),
            )
            name = well_system.names[idx]
            if name in reresolved:
                reresolved.remove(name)
            reresolved.append(name)

    return tuple(reresolved)
