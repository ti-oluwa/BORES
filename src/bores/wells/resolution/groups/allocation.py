"""
`CompiledGroupControls` target allocation. Splits a group's target rate
across its eligible member wells by guide rate, writing concrete per-well
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

import numpy as np

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
from bores.wells.hydraulics.base import SurfaceFluidProperties, WellBoreModel
from bores.wells.resolution.engine import resolve_well_control
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.states import ConnectionSample
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

    group_index = group_controls.group_row(name=group_name)

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
        raise ValidationError(f"Group {group_name!r}'s control has no `target_rate`.")

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


def get_group_controlled_rate(
    *, well_system: CompiledWellSystem, workspace: WellsWorkspace, well_row: Integer
) -> Number:
    """
    A member well's own resolved rate, in whatever quantity and
    condition (surface or reservoir) its current concrete control mode
    actually targets - the one value directly comparable against
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
            phase_array = {
                FluidPhaseTag.OIL: workspace.surface_oil_rates,
                FluidPhaseTag.WATER: workspace.surface_water_rates,
                FluidPhaseTag.GAS: workspace.surface_gas_rates,
            }[FluidPhaseTag(controls.injected_phases[well_row])]
            return phase_array[well_row]
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


def redistribute_group_shortfall(
    *,
    group_name: str,
    allocated_wells: typing.Sequence[str],
    well_system: CompiledWellSystem,
    workspace: WellsWorkspace,
    control_spec: WellControlSpec,
    get_wellbore: typing.Callable[[Integer], WellBoreModel],
    get_connection_samples: typing.Callable[[Integer], typing.Sequence[ConnectionSample]],
    get_surface_fluid_properties: typing.Callable[[Integer], SurfaceFluidProperties | None]
    | None = None,
) -> tuple[str, ...]:
    """
    Redistributes a group's shortfall across members with headroom.

    `allocate_group_targets` only does a first-pass proportional split.
    A member that can't actually reach its own share (its own BHP floor
    binds it first, say) leaves the group short of its real target, with
    nothing correcting for it. This looks for exactly that: a bound
    member has its own target capped down to what it actually delivered
    (so the same gap isn't counted again next pass), and its unmet share
    is added to this pass's total shortfall. Any member already sitting
    at or above its own target has headroom (nothing bound it below what
    it was asked for), so its target grows by its guide-rate share of
    that shortfall, and it's re-resolved. Repeats, rechecking every
    member each pass, until the shortfall clears, no member has headroom
    left, or `control_spec.max_fixed_point_iterations` passes are used.

    Requires `allocate_group_targets` and an initial resolve of every
    member in `allocated_wells` to have already run this timestep - reads
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
    :param get_wellbore: Given a well row, its hydraulics correlation.
        Only called for a member actually being re-resolved.
    :param get_connection_samples: Given a well row, its active, open
        connections' current reservoir samples. Only called for a member
        actually being re-resolved.
    :param get_surface_fluid_properties: Given a well row, its surface
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
    # Resolve group_name up front, purely to validate it, mirroring every
    # other function in this module - group_row itself is unused below,
    # since eligibility comes from allocated_wells, not group membership.
    group_controls.group_row(name=group_name)

    if not allocated_wells:
        return ()

    controls = well_system.controls
    grup_derived = [well_system.well_row(name=name) for name in allocated_wells]

    resolved_get_surface_fluid_properties = get_surface_fluid_properties or (lambda well_row: None)
    reresolved: list[str] = []

    for _ in range(control_spec.max_fixed_point_iterations):
        headroom_rows: list[Integer] = []
        total_shortfall = 0.0
        for i in grup_derived:
            target = controls.target_rates[i]
            if math.isnan(target):
                continue
            actual = get_group_controlled_rate(
                well_system=well_system, workspace=workspace, well_row=i
            )
            gap = target - actual
            tolerance = (
                abs(target) * control_spec.rate_convergence_tolerance
                if target != 0
                else control_spec.rate_convergence_tolerance
            )
            if gap > tolerance:
                # This well is bound below its own target (its BHP floor,
                # say) - cap its target down to what it actually
                # delivered, so this same gap isn't counted again next
                # pass. Its unmet share becomes this pass's shortfall to
                # redistribute, once, not every remaining iteration.
                total_shortfall += gap
                controls.target_rates[i] = actual
            elif gap <= tolerance:
                headroom_rows.append(i)

        if total_shortfall <= 0 or not headroom_rows:
            break

        guide_rates = controls.guide_rates[np.asarray(headroom_rows, dtype=np.int64)]
        weights = np.where(np.isnan(guide_rates), 1.0, guide_rates)
        total_weight = weights.sum()

        for i, weight in zip(headroom_rows, weights, strict=False):
            controls.target_rates[i] += total_shortfall * (weight / total_weight)
            resolve_well_control(
                compiled_system=well_system,
                well_row=i,
                wellbore=get_wellbore(i),
                connection_samples=get_connection_samples(i),
                workspace=workspace,
                control_spec=control_spec,
                surface_fluid_properties=resolved_get_surface_fluid_properties(i),
            )
            name = well_system.names[i]
            if name in reresolved:
                reresolved.remove(name)
            reresolved.append(name)

    return tuple(reresolved)
