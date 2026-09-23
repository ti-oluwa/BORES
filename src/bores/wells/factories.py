"""Convenience constructors for building wells directly, without a deck."""

import typing

from bores.errors import ValidationError
from bores.types import FluidPhase, Number, Orientation, UnitSystem
from bores.wells.base import (
    Perforation,
    Well,
    Wells,
    WellStatus,
    WellType,
)
from bores.wells.controls import (
    BHPLimit,
    InjectorControl,
    InjectorControlMode,
    Limit,
    ProducerControl,
    ProducerControlMode,
    RateLimit,
    RateQuantity,
    WellControl,
    WellControls,
)
from bores.wells.groups import (
    GroupControl,
    GroupInjectorControlMode,
    GroupProducerControlMode,
)
from bores.wells.trajectory import WellTrajectory

__all__ = [
    "make_group_control",
    "make_injector",
    "make_producer",
    "make_well_controls",
    "make_wells",
]


QUANTITY_PRODUCER_CONTROL_MODES = {
    RateQuantity.OIL: ProducerControlMode.OIL_RATE,
    RateQuantity.WATER: ProducerControlMode.WATER_RATE,
    RateQuantity.GAS: ProducerControlMode.GAS_RATE,
    RateQuantity.LIQUID: ProducerControlMode.LIQUID_RATE,
    RateQuantity.RESERVOIR: ProducerControlMode.RESERVOIR_VOLUME_RATE,
}

PHASE_RATE_QUANTITY = {
    FluidPhase.OIL: RateQuantity.OIL,
    FluidPhase.WATER: RateQuantity.WATER,
    FluidPhase.GAS: RateQuantity.GAS,
}


def _resolve_perforations(
    *,
    perforation_depths: tuple[Number, Number] | None,
    perforations: typing.Sequence[Perforation] | None,
    reference_depth: Number | None,
    wellbore_radius: Number,
    skin: Number,
    saturation_region: int | None,
    connection_factor_override: Number | None,
    connection_factor_multiplier: Number | None,
    direction: Orientation | None,
    schedule_status: WellStatus,
) -> tuple[tuple[Perforation, ...], Number]:
    """
    Resolves `make_producer`/`make_injector`'s two perforation-building
    paths into one `(perforations, reference_depth)` pair.

    **Simple path** (`perforation_depths`): builds one vertical `Perforation`
    from the flat per-perforation params (`wellbore_radius`, `skin`, etc.) -
    today's only option, unchanged.

    **Direct path** (`perforations`): passes a caller-built sequence of
    `Perforation` straight through. Multiple intervals, or (paired with
    `trajectory=` on the well itself) intervals with `top_md`/`bottom_md`
    set for a deviated well. The flat per-perforation params don't apply
    here (each interval already carries its own); `reference_depth` has no
    safe default across multiple intervals or a deviated trajectory, so
    it's required in this path.

    `Well`'s post initialization logic is what actually enforces the
    `top_md`/`bottom_md` needs `trajectory` (and vice versa) rule, once the
    resolved perforations reach `Well(...)`.

    :returns: `(perforations, resolved_reference_depth)`.
    :raises ValidationError: If both or neither of `perforation_depths`/
        `perforations` is given, or `perforations` is given without `reference_depth`.
    """
    if (perforation_depths is None) == (perforations is None):
        raise ValidationError(
            "Supply exactly one of `perforation_depths` (single vertical "
            "perforation) or `perforations` (full "
            "control: multiple intervals, or `top_md`/`bottom_md`-based intervals for a "
            "deviated well via `trajectory=`)."
        )

    if perforations is not None:
        if reference_depth is None:
            raise ValidationError(
                "`reference_depth` is required when `perforations` is given "
                "directly. There's no single safe default across multiple "
                "intervals or a deviated trajectory."
            )
        return tuple(perforations), reference_depth

    top_depth, bottom_depth = typing.cast(tuple[Number, Number], perforation_depths)
    resolved_reference_depth = reference_depth if reference_depth is not None else bottom_depth
    perforation = Perforation(
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        wellbore_radius=wellbore_radius,
        skin=skin,
        saturation_region=saturation_region,
        connection_factor_override=connection_factor_override,
        connection_factor_multiplier=connection_factor_multiplier,
        direction=direction,
        schedule_status=schedule_status,
    )
    return (perforation,), resolved_reference_depth


def make_producer(
    name: str,
    *,
    surface_location: tuple[Number, Number],
    perforation_depths: tuple[Number, Number] | None = None,
    perforations: typing.Sequence[Perforation] | None = None,
    trajectory: WellTrajectory | None = None,
    reference_depth: Number | None = None,
    target_rate: Number | None = None,
    rate_quantity: RateQuantity = RateQuantity.OIL,
    target_bhp: Number | None = None,
    target_thp: Number | None = None,
    min_bhp: Number | None = None,
    max_rate: Number | None = None,
    limits: typing.Sequence[Limit] | None = None,
    efficiency_factor: Number = 1.0,
    guide_rate: Number | None = None,
    wellbore_radius: Number = 0.25,
    skin: Number = 0.0,
    saturation_region: int | None = None,
    connection_factor_override: Number | None = None,
    connection_factor_multiplier: Number | None = None,
    direction: Orientation | None = None,
    preferred_phase: FluidPhase | None = None,
    pvt_region: int | None = None,
    group: str | None = None,
    schedule_status: WellStatus = WellStatus.ACTIVE,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Mapping[str, typing.Any] | None = None,
) -> tuple[Well, ProducerControl]:
    """
    Builds a producer `Well` and its `ProducerControl`, for direct
    (non-deck) API use.

    Control mode is chosen by priority: `target_bhp`, then `target_thp`,
    then `target_rate` at `rate_quantity`.

    :param name: Well name.
    :param surface_location: `(x, y)` wellhead location.
    :param perforation_depths: `(top_depth, bottom_depth)` for one single
        vertical perforation. Mutually exclusive with `perforations`;
        supply exactly one.
    :param perforations: A caller-built sequence of `Perforation`,
        for multiple intervals or (with `trajectory=`) a deviated well,
        with `top_md`/`bottom_md` set instead of `top_depth`/`bottom_depth`.
        `reference_depth` is required in this path. Mutually exclusive
        with `perforation_depths`.
    :param trajectory: Deviation survey. Only meaningful with `perforations`
        that have `top_md`/`bottom_md` set. See `Well.trajectory`.
    :param reference_depth: BHP/THP reporting datum. Defaults to
        `perforation_depths[1]` in the simple path; required in the direct path.
    :param target_rate: Rate target.
    :param rate_quantity: Which rate `target_rate` is.
    :param target_bhp: BHP target.
    :param target_thp: THP target.
    :param min_bhp: If given, adds a `BHPLimit(min_value=min_bhp)`.
    :param max_rate: If given, adds a `RateLimit` on `rate_quantity` capped at `max_rate`.
    :param limits: Additional limits, appended after any built from `min_bhp`/`max_rate`.
    :param efficiency_factor: `ProducerControl.efficiency_factor`.
    :param guide_rate: `ProducerControl.guide_rate`.
    :param wellbore_radius: `Perforation.wellbore_radius`.
    :param skin: `Perforation.skin`.
    :param saturation_region: `Perforation.saturation_region`.
    :param connection_factor_override: `Perforation.connection_factor_override`.
    :param connection_factor_multiplier: `Perforation.connection_factor_multiplier`.
    :param direction: `Perforation.direction`.
    :param preferred_phase: `Well.preferred_phase`.
    :param pvt_region: `Well.pvt_region`.
    :param group: `Well.group`.
    :param schedule_status: `Well.schedule_status`, and the simple path's
        single `Perforation.schedule_status`.
    :param unit_system: `Well.unit_system` and `ProducerControl.unit_system`.
    :param metadata: `Well.metadata`.
    :returns: `(Well, ProducerControl)`.
    :raises ValidationError: If none of `target_rate`, `target_bhp`, or
        `target_thp` is given; if both or neither of `perforation_depths`/
        `perforations` is given; or if `perforations` is given without `reference_depth`.
    """
    if target_rate is None and target_bhp is None and target_thp is None:
        raise ValidationError("Supply one of `target_rate`, `target_bhp`, or `target_thp`.")

    resolved_perforations, resolved_reference_depth = _resolve_perforations(
        perforation_depths=perforation_depths,
        perforations=perforations,
        reference_depth=reference_depth,
        wellbore_radius=wellbore_radius,
        skin=skin,
        saturation_region=saturation_region,
        connection_factor_override=connection_factor_override,
        connection_factor_multiplier=connection_factor_multiplier,
        direction=direction,
        schedule_status=schedule_status,
    )
    well = Well(
        name=name,
        well_type=WellType.PRODUCER,
        surface_location=surface_location,
        reference_depth=resolved_reference_depth,
        perforations=resolved_perforations,
        trajectory=trajectory,
        preferred_phase=preferred_phase,
        pvt_region=pvt_region,
        group=group,
        schedule_status=schedule_status,
        unit_system=unit_system,
        metadata=metadata,
    )

    resolved_limits: list[Limit] = []
    if min_bhp is not None:
        resolved_limits.append(BHPLimit(min_value=min_bhp))
    if max_rate is not None:
        resolved_limits.append(RateLimit(quantity=rate_quantity, max_value=max_rate))
    if limits:
        resolved_limits.extend(limits)

    if target_bhp is not None:
        control = ProducerControl(
            mode=ProducerControlMode.BHP,
            target_bhp=target_bhp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    elif target_thp is not None:
        control = ProducerControl(
            mode=ProducerControlMode.THP,
            target_thp=target_thp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    else:
        control = ProducerControl(
            mode=QUANTITY_PRODUCER_CONTROL_MODES[rate_quantity],
            target_rate=target_rate,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    return well, control


def make_injector(
    name: str,
    *,
    injected_phase: FluidPhase,
    surface_location: tuple[Number, Number],
    perforation_depths: tuple[Number, Number] | None = None,
    perforations: typing.Sequence[Perforation] | None = None,
    trajectory: WellTrajectory | None = None,
    reference_depth: Number | None = None,
    target_rate: Number | None = None,
    target_bhp: Number | None = None,
    target_thp: Number | None = None,
    max_bhp: Number | None = None,
    max_rate: Number | None = None,
    limits: typing.Sequence[Limit] | None = None,
    efficiency_factor: Number = 1.0,
    guide_rate: Number | None = None,
    wellbore_radius: Number = 0.25,
    skin: Number = 0.0,
    saturation_region: int | None = None,
    connection_factor_override: Number | None = None,
    connection_factor_multiplier: Number | None = None,
    direction: Orientation | None = None,
    pvt_region: int | None = None,
    group: str | None = None,
    schedule_status: WellStatus = WellStatus.ACTIVE,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Mapping[str, typing.Any] | None = None,
) -> tuple[Well, InjectorControl]:
    """
    Builds an injector `Well` and its `InjectorControl`, for direct
    (non-deck) API use.

    Control mode is chosen by priority: `target_bhp`, then `target_thp`,
    then `target_rate`.

    :param name: Well name.
    :param injected_phase: Phase being injected.
    :param surface_location: `(x, y)` wellhead location.
    :param perforation_depths: `(top_depth, bottom_depth)` for one single
        vertical perforation. Mutually exclusive with `perforations`;
        supply exactly one.
    :param perforations: A caller-built sequence of `Perforation`,
        for multiple intervals or (with `trajectory=`) a deviated well,
        with `top_md`/`bottom_md` set instead of `top_depth`/`bottom_depth`.
        `reference_depth` is required in this path. Mutually exclusive
        with `perforation_depths`.
    :param trajectory: Deviation survey. Only meaningful with `perforations`
        that have `top_md`/`bottom_md` set. See `Well.trajectory`.
    :param reference_depth: BHP/THP reporting datum. Defaults to
        `perforation_depths[1]` in the simple path; required in the direct path.
    :param target_rate: Rate target.
    :param target_bhp: BHP target.
    :param target_thp: THP target.
    :param max_bhp: If given, adds a `BHPLimit(max_value=max_bhp)`.
    :param max_rate: If given, adds a `RateLimit` on `injected_phase` capped at `max_rate`.
    :param limits: Additional limits, appended after any built from `max_bhp`/`max_rate`.
    :param efficiency_factor: `InjectorControl.efficiency_factor`.
    :param guide_rate: `InjectorControl.guide_rate`.
    :param wellbore_radius: `Perforation.wellbore_radius`.
    :param skin: `Perforation.skin`.
    :param saturation_region: `Perforation.saturation_region`.
    :param connection_factor_override: `Perforation.connection_factor_override`.
    :param connection_factor_multiplier: `Perforation.connection_factor_multiplier`.
    :param direction: `Perforation.direction`.
    :param pvt_region: `Well.pvt_region`.
    :param group: `Well.group`.
    :param schedule_status: `Well.schedule_status`, and the simple path's
        single `Perforation.schedule_status`.
    :param unit_system: `Well.unit_system` and `InjectorControl.unit_system`.
    :param metadata: `Well.metadata`.
    :returns: `(Well, InjectorControl)`.
    :raises ValidationError: If none of `target_rate`, `target_bhp`, or
        `target_thp` is given; if both or neither of `perforation_depths`/
        `perforations` is given; or if `perforations` is given without `reference_depth`.
    """
    if target_rate is None and target_bhp is None and target_thp is None:
        raise ValidationError("Supply one of `target_rate`, `target_bhp`, or `target_thp`.")

    resolved_perforations, resolved_reference_depth = _resolve_perforations(
        perforation_depths=perforation_depths,
        perforations=perforations,
        reference_depth=reference_depth,
        wellbore_radius=wellbore_radius,
        skin=skin,
        saturation_region=saturation_region,
        connection_factor_override=connection_factor_override,
        connection_factor_multiplier=connection_factor_multiplier,
        direction=direction,
        schedule_status=schedule_status,
    )
    well = Well(
        name=name,
        well_type=WellType.INJECTOR,
        surface_location=surface_location,
        reference_depth=resolved_reference_depth,
        perforations=resolved_perforations,
        trajectory=trajectory,
        preferred_phase=injected_phase,
        pvt_region=pvt_region,
        group=group,
        schedule_status=schedule_status,
        unit_system=unit_system,
        metadata=metadata,
    )

    resolved_limits: list[Limit] = []
    if max_bhp is not None:
        resolved_limits.append(BHPLimit(max_value=max_bhp))
    if max_rate is not None:
        resolved_limits.append(
            RateLimit(quantity=PHASE_RATE_QUANTITY[injected_phase], max_value=max_rate)
        )
    if limits:
        resolved_limits.extend(limits)

    if target_bhp is not None:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.BHP,
            target_bhp=target_bhp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    elif target_thp is not None:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.THP,
            target_thp=target_thp,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    else:
        control = InjectorControl(
            injected_phase=injected_phase,
            mode=InjectorControlMode.RATE,
            target_rate=target_rate,
            limits=tuple(resolved_limits),
            efficiency_factor=efficiency_factor,
            guide_rate=guide_rate,
            unit_system=unit_system,
        )
    return well, control


QUANTITY_GROUP_CONTROL_MODES = {
    RateQuantity.OIL: GroupProducerControlMode.OIL_RATE,
    RateQuantity.WATER: GroupProducerControlMode.WATER_RATE,
    RateQuantity.GAS: GroupProducerControlMode.GAS_RATE,
    RateQuantity.LIQUID: GroupProducerControlMode.LIQUID_RATE,
    RateQuantity.RESERVOIR: GroupProducerControlMode.RESERVOIR_VOLUME_RATE,
}


def make_group_control(
    *,
    target_rate: Number,
    quantity: RateQuantity = RateQuantity.OIL,
    injected_phase: FluidPhase | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
) -> GroupControl:
    """
    Builds a `GroupControl`.

    :param target_rate: Group rate target.
    :param quantity: Which rate `target_rate` is. Ignored if `injected_phase` is given.
    :param injected_phase: If given, builds an injection `GroupControl`
        (`mode=GroupInjectorControlMode.RATE`) instead of a production one.
    :param unit_system: `GroupControl.unit_system`.
    :returns: Constructed `GroupControl`.
    """
    if injected_phase is not None:
        return GroupControl(
            mode=GroupInjectorControlMode.RATE,
            target_rate=target_rate,
            injected_phase=injected_phase,
            unit_system=unit_system,
        )
    return GroupControl(
        mode=QUANTITY_GROUP_CONTROL_MODES[quantity],
        target_rate=target_rate,
        unit_system=unit_system,
    )


def make_wells(*wells: Well) -> Wells:
    """
    Builds a `Wells` from a set of `Well` objects.

    :param wells: `Well` objects to include, each already carrying its own name.
    :returns: `Wells`, keyed by each well's own `.name`.
    """
    return Wells(wells={well.name: well for well in wells})


def make_well_controls(**controls: WellControl) -> WellControls:
    """
    Builds a `WellControls` from a set of named controls.

    :param controls: `well_name=control` keyword pairs.
    :returns: `WellControls`, keyed by the given names.
    """
    return WellControls(controls=dict(controls))
