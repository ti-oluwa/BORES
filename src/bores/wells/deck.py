"""
Utilities for building well model definition objects from parsed Eclipse deck records.
"""

import typing

import attrs

from bores.constants import c
from bores.deck.core import DeckParseError
from bores.deck.file import DeckFile
from bores.errors import NotSupportedError, ValidationError
from bores.grids.base import Grid
from bores.schedule.base import Rule, Schedule
from bores.schedule.events import TimeEvent
from bores.types import FluidPhase, UnitSystem
from bores.wells.base import CompletionStatus, Perforation, Well, Wells, WellStatus, WellType
from bores.wells.compile import LimitKind
from bores.wells.controls import (
    BHPLimit,
    EconomicLimit,
    EconomicQuantity,
    InjectorControl,
    InjectorControlMode,
    Limit,
    ProducerControl,
    ProducerControlMode,
    THPLimit,
    WellControl,
    WellControls,
    WorkoverAction,
)
from bores.wells.groups import (
    GroupControl,
    GroupControls,
    WellGroup,
    WellGroups,
)
from bores.wells.mappings import (
    DIRECTION_MAP,
    ECONOMIC_MIN_RATE_QUANTITY_FIELDS,
    ECONOMIC_QUANTITY_FIELDS,
    GROUP_INJECTOR_CONTROL_MODE_MAP,
    GROUP_PRODUCER_CONTROL_MODE_MAP,
    INJECTOR_CONTROL_MODE_MAP,
    PRODUCER_CONTROL_MODE_MAP,
    WELOPEN_STATUS_MAP,
    WELTARG_TARGET_FIELD,
)
from bores.wells.schedule import (
    ActivateCompletion,
    ActivateWell,
    MultiplyConnectionFactor,
    OpenWell,
    SetLimit,
    SetWellControl,
    SetWellTarget,
)

# NOTE: Future Self, All imports from `wells.deck` in other modules
# (mostly in the `wells.*`) should be inline. Top-level imports will cause
# circular import issues. Also, avoid doing a top-level import of `blackoil.*`
# in the `wells` module
if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel

__all__ = [
    "apply_d_factors",
    "apply_economic_limits",
    "apply_guide_rates",
    "from_deck_gas_rate",
    "load_controls_from_records",
    "load_economic_limits_from_record",
    "load_group_control_from_record",
    "load_group_controls",
    "load_group_controls_from_records",
    "load_groups",
    "load_groups_from_records",
    "load_injector_control_from_record",
    "load_producer_control_from_record",
    "load_schedule",
    "load_well_controls",
    "load_well_from_records",
    "load_wells",
    "load_wells_from_records",
    "select_current_records",
]


def load_well_from_records(
    grid: Grid,
    welspecs_record: typing.Mapping[str, typing.Any],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    welopen_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    well_type: WellType = WellType.PRODUCER,
    current_time: float = 0.0,
) -> Well:
    """
    Builds one well from its `WELSPECS` record and its `COMPDAT` records.

    Every `COMPDAT` record for the well is included, no matter when in the
    schedule it takes effect. A completion whose scheduled time has not
    been reached yet is included but marked as not yet active, so a
    workover completion added later in the schedule is already part of the
    well from the start and just needs switching on when its time comes,
    rather than being added on the fly. The well itself is marked the same
    way, based on when its `WELSPECS` record takes effect.

    Any `WELOPEN` record for the well, up to `current_time`, is replayed
    in schedule order afterwards, so a completion (or the whole well)
    opened or shut later in the schedule without a fresh `COMPDAT` still
    ends up with the right status.

    :param grid: Grid the well's completions are resolved against.
    :param welspecs_record: The well's `WELSPECS` record.
    :param compdat_records: Every `COMPDAT` record for this well, from any
        point in the schedule.
    :param wpimult_records: Every `WPIMULT` record for this well, if any.
    :param welopen_records: Every `WELOPEN` record for this well, if any.
    :param unit_system: The deck's unit system.
    :param well_type: Whether this well is a producer or an injector. Not
        derivable from `WELSPECS`/`COMPDAT` alone.
    :param current_time: The point on the schedule clock this well is
        being built for, in the deck's time unit. Anything scheduled at or
        before this time is marked active; anything later is marked
        pending. Defaults to zero, the start of the run.
    :returns: The constructed well.
    :raises ValidationError: If there are no `COMPDAT` records for this well.
    """
    if not compdat_records:
        raise ValidationError(f"No `COMPDAT` records for well {welspecs_record['well']!r}.")

    perforation_specs = []
    dims = grid.dimensions
    if dims is None:
        raise ValidationError(
            "Cannot ascertain grid dimensions. Ensure that the provided `Grid` has `dimensions`."
        )

    if welspecs_record["inflow_equation"] != "STD":
        raise NotSupportedError(
            "Only the standard inflow equation ('STD') is currently supported."
        )

    whole_well_multiplier: float | None = None
    multiplier_by_ijk: dict[tuple[int, int, int, int], float] = {}
    if wpimult_records:
        for record in wpimult_records:
            i, j = record.get("i", 0), record.get("j", 0)
            k1, k2 = record.get("k1", 0), record.get("k2", 0)
            if i == 0 and j == 0 and k1 == 0 and k2 == 0:
                # `WPIMULT`'s own schema defaults I/J/K1/K2 to 0, meaning
                # "every connection on this well", that key would never
                # match any COMPDAT's real (nonzero) indices below, so a
                # whole-well multiplier (the common case) needs its own
                # fallback slot rather than living in multiplier_by_ijk.
                # Later `WPIMULT` reissues overwrite earlier ones, matching
                # this file's other reissue semantics (`WCONPROD`/`WCONINJE`).
                whole_well_multiplier = record["multiplier"]
            else:
                multiplier_by_ijk[i, j, k1, k2] = record["multiplier"]

    for record in compdat_records:
        # Minus 1, to move from 1-based to 0-based indexing used internally
        i, j = record["i"], record["j"]
        k1, k2 = record["k1"], record["k2"]
        top_cell = dims.flat_index(i - 1, j - 1, k1 - 1)
        bottom_cell = dims.flat_index(i - 1, j - 1, k2 - 1)
        top_depth = grid.cell_min_xyz[top_cell, 2]
        bottom_depth = grid.cell_max_xyz[bottom_cell, 2]
        multiplier_key = (i, j, k1, k2)
        direction = record.get("direction")
        saturation_region = record.get("saturation_table") or None  # 0 should map to None too
        radius = (record.get("diameter") or 0) * 0.5
        skin = record.get("skin", 0.0) or 0.0
        status = (
            CompletionStatus.OPEN
            if record.get("status", "OPEN") == "OPEN"
            else CompletionStatus.SHUT
        )
        schedule_status = (
            WellStatus.ACTIVE
            if record.get("schedule_time", 0.0) <= current_time
            else WellStatus.PENDING
        )
        perforation_specs.append({
            "i": i,
            "j": j,
            "k1": k1,
            "k2": k2,
            "top_depth": top_depth,
            "bottom_depth": bottom_depth,
            "skin": skin,
            "wellbore_radius": radius,
            "status": status,
            "saturation_region": saturation_region,
            "connection_factor_override": record.get("connection_factor"),
            "connection_factor_multiplier": multiplier_by_ijk.get(
                multiplier_key, whole_well_multiplier
            ),
            "direction": DIRECTION_MAP.get(direction) if direction else None,
            "schedule_status": schedule_status,
        })

    if welopen_records:
        # Whole-well or per-connection open/shut events, replayed in
        # schedule order up to current_time. A whole-well event
        # (i=j=k1=k2=0) touches every completion; a targeted one only
        # touches completions whose own (i, j) column and k-layer range
        # overlap the event's k1-k2 range.
        for record in sorted(
            (r for r in welopen_records if r.get("schedule_time", 0.0) <= current_time),
            key=lambda r: r.get("schedule_time", 0.0),
        ):
            new_status = WELOPEN_STATUS_MAP[record["status"]]
            i, j = record.get("i", 0), record.get("j", 0)
            k1, k2 = record.get("k1", 0), record.get("k2", 0)
            whole_well = i == 0 and j == 0 and k1 == 0 and k2 == 0
            for spec in perforation_specs:
                if whole_well or (
                    spec["i"] == i and spec["j"] == j and spec["k1"] <= k2 and spec["k2"] >= k1
                ):
                    spec["status"] = new_status

    perforations = [
        Perforation(
            top_depth=spec["top_depth"],
            bottom_depth=spec["bottom_depth"],
            skin=spec["skin"],
            wellbore_radius=spec["wellbore_radius"],
            status=spec["status"],
            saturation_region=spec["saturation_region"],
            connection_factor_override=spec["connection_factor_override"],
            connection_factor_multiplier=spec["connection_factor_multiplier"],
            direction=spec["direction"],
            schedule_status=spec["schedule_status"],
        )
        for spec in perforation_specs
    ]

    reference_depth = welspecs_record.get("reference_depth")
    deepest_bottom_depth = max(perforation.bottom_depth for perforation in perforations)
    surface_location = grid.get_cell_center_at(
        welspecs_record["i"] - 1,
        welspecs_record["j"] - 1,
        0,  # At surface
    )[:2]
    pvt_region = welspecs_record.get("pvt_table") or None  # 0 should map to None too
    preferred_phase = (
        FluidPhase(welspecs_record["phase"].lower()) if welspecs_record.get("phase") else None
    )
    well_schedule_status = (
        WellStatus.ACTIVE
        if welspecs_record.get("schedule_time", 0.0) <= current_time
        else WellStatus.PENDING
    )
    return Well(
        name=welspecs_record["well"],
        well_type=well_type,
        surface_location=surface_location,
        reference_depth=reference_depth if reference_depth is not None else deepest_bottom_depth,
        perforations=tuple(perforations),
        preferred_phase=preferred_phase,
        group=welspecs_record.get("group"),
        pvt_region=pvt_region,
        unit_system=unit_system,
        schedule_status=well_schedule_status,
    )


def load_wells_from_records(
    grid: Grid,
    welspecs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    welopen_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    injector_names: typing.Container[str] = (),
    current_time: float = 0.0,
) -> Wells:
    """
    Builds a full well roster from every `WELSPECS` and `COMPDAT` record in a
    deck, no matter where in the schedule each one occurs.

    A well introduced later in the schedule, or a completion added to an
    existing well later (a workover, for example), is still built now and
    included in the roster, just marked as not yet active. This means the
    whole run's wells and completions only need building once. As the
    schedule reaches each one's time, it can be switched on in place
    instead of being added partway through the run.

    :param grid: Grid the wells' completions are resolved against.
    :param welspecs_records: Every `WELSPECS` record in the deck.
    :param compdat_records: Every `COMPDAT` record in the deck.
    :param wpimult_records: Every `WPIMULT` record in the deck, if any.
    :param welopen_records: Every `WELOPEN` record in the deck, if any.
    :param unit_system: The deck's unit system.
    :param injector_names: Names of wells that appear in `WCONINJE`. Every
        other well is built as a producer, since `WELSPECS` and `COMPDAT`
        alone don't say which a well is.
    :param current_time: The point on the schedule clock the roster is
        being built for, in the deck's time unit. Anything scheduled at or
        before this time is marked active; anything later is marked
        pending. Defaults to zero, the start of the run.
    :returns: Wells keyed by name, covering the well's whole life in the schedule.
    """
    compdat_by_well: dict[str, list[typing.Mapping[str, typing.Any]]] = {}
    wpimult_by_well: dict[str, list[typing.Mapping[str, typing.Any]]] = {}
    welopen_by_well: dict[str, list[typing.Mapping[str, typing.Any]]] = {}
    for record in compdat_records:
        compdat_by_well.setdefault(record["well"], []).append(record)

    if wpimult_records:
        for record in wpimult_records:
            wpimult_by_well.setdefault(record["well"], []).append(record)

    if welopen_records:
        for record in welopen_records:
            welopen_by_well.setdefault(record["well"], []).append(record)

    wells = {
        record["well"]: load_well_from_records(
            grid,
            welspecs_record=record,
            compdat_records=compdat_by_well.get(record["well"], []),
            wpimult_records=wpimult_by_well.get(record["well"]),
            welopen_records=welopen_by_well.get(record["well"]),
            well_type=(
                WellType.INJECTOR if record["well"] in injector_names else WellType.PRODUCER
            ),
            unit_system=unit_system,
            current_time=current_time,
        )
        for record in welspecs_records
    }
    return Wells(wells=wells)


def from_deck_gas_rate(value: float | None, unit_system: UnitSystem) -> float | None:
    """
    :param value: Raw gas rate value as written in the deck, or None.
    :param unit_system: The deck's unit system.
    :returns: value unchanged, except multiplied by 1000 when
        `unit_system` is FIELD. Eclipse's FIELD convention reports gas
        rates in Mscf/day; this codebase's internal FIELD convention is
        raw scf/day, dimensionally consistent with oil/water in stb/day.
        Not applied for any other `unit_system`.
    """
    if value is None:
        return None
    return value * c.MSCF_TO_SCF if unit_system is UnitSystem.FIELD else value


def select_current_records(
    records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    key: str,
    current_time: float,
) -> dict[str, typing.Mapping[str, typing.Any]]:
    """
    Picks, for each well or group name, whichever one record from a
    keyword is actually in effect at a given point in the schedule.

    A record scheduled for later than the given time is not counted yet.
    Among the records that have already taken effect for a name, the one
    scheduled most recently wins, since it is the latest change.

    :param records: Every record for one keyword, from any point in the schedule.
    :param key: Which field on a record holds the well or group name.
    :param current_time: The point on the schedule clock to resolve records for.
    :returns: One record per name, the one currently in effect.
    """
    current: dict[str, typing.Mapping[str, typing.Any]] = {}
    current_times: dict[str, float] = {}
    for record in records:
        schedule_time = record.get("schedule_time", 0.0)
        if schedule_time > current_time:
            continue
        name = record[key]
        if name not in current or schedule_time >= current_times[name]:
            current[name] = record
            current_times[name] = schedule_time
    return current


def load_producer_control_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> ProducerControl:
    """
    Build a `ProducerControl` from one `WCONPROD` record.

    :param record: One parsed `WCONPROD` record.
    :param unit_system: The deck's unit system.
    :returns: Constructed `ProducerControl`. Adds an implicit `BHPLimit(min_value=bhp)`
        when `bhp` is given and mode isn't `BHP`, and an implicit
        `THPLimit(min_value=thp)` when `thp` is given and mode isn't `THP`.
    """
    mode = PRODUCER_CONTROL_MODE_MAP[record["control_mode"]]
    limits: list[Limit] = []
    bhp = record.get("bhp")
    if bhp is not None and mode is not ProducerControlMode.BHP:
        limits.append(BHPLimit(min_value=bhp, unit_system=unit_system))

    thp = record.get("thp")
    if thp is not None and mode is not ProducerControlMode.THP:
        limits.append(THPLimit(min_value=thp, unit_system=unit_system))

    if mode is ProducerControlMode.OIL_RATE:
        target_rate = record.get("oil_rate")
    elif mode is ProducerControlMode.WATER_RATE:
        target_rate = record.get("water_rate")
    elif mode is ProducerControlMode.GAS_RATE:
        target_rate = from_deck_gas_rate(record.get("gas_rate"), unit_system)
    elif mode is ProducerControlMode.LIQUID_RATE:
        target_rate = record.get("liquid_rate")
    elif mode is ProducerControlMode.RESERVOIR_VOLUME_RATE:
        target_rate = record.get("reservoir_volume_rate")
    else:
        target_rate = None

    return ProducerControl(
        mode=mode,
        target_rate=target_rate,
        target_bhp=bhp,
        target_thp=record.get("thp"),
        limits=tuple(limits),
        unit_system=unit_system,
    )


def load_injector_control_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> InjectorControl:
    """
    Build an `InjectorControl` from one `WCONINJE` record.

    :param record: One parsed `WCONINJE` record.
    :param unit_system: The deck's unit system.
    :returns: Constructed `InjectorControl`. Adds an implicit `BHPLimit(max_value=bhp)`
        when `bhp` is given and mode isn't `BHP`, and an implicit
        `THPLimit(max_value=thp)` when `thp` is given and mode isn't `THP`.
    """
    mode = INJECTOR_CONTROL_MODE_MAP[record["control_mode"]]
    phase = FluidPhase(record["injector_type"].lower())
    limits: list[Limit] = []
    bhp = record.get("bhp")
    if bhp is not None and mode is not InjectorControlMode.BHP:
        limits.append(BHPLimit(max_value=bhp, unit_system=unit_system))

    thp = record.get("thp")
    if thp is not None and mode is not InjectorControlMode.THP:
        limits.append(THPLimit(max_value=thp, unit_system=unit_system))

    rate = record.get("rate")
    if phase is FluidPhase.GAS:
        rate = from_deck_gas_rate(rate, unit_system)

    return InjectorControl(
        injected_phase=phase,
        mode=mode,
        target_rate=rate,
        target_bhp=bhp,
        target_thp=record.get("thp"),
        limits=tuple(limits),
        unit_system=unit_system,
    )


def load_economic_limits_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> tuple[EconomicLimit, ...]:
    """
    Loads a well's economic limits from one `WECON` record.

    :param record: One parsed `WECON` record.
    :param unit_system: The deck's unit system.
    :returns: One economic limit per non-zero threshold present on the
        record (minimum oil rate, water cut, GOR, water-gas ratio). A
        minimum oil rate of exactly zero is treated as "no limit", matching
        the deck's own default.
    """
    workover_action = WorkoverAction(record.get("workover_action", "WELL"))
    end_run = bool(record.get("end_run", False))

    limits = []
    for quantity, field_name in ECONOMIC_MIN_RATE_QUANTITY_FIELDS.items():
        value = record.get(field_name)
        if not value:
            continue
        limits.append(
            EconomicLimit(
                quantity=quantity,
                min_value=value,
                workover_action=workover_action,
                end_run=end_run,
                unit_system=unit_system,
            )
        )

    for quantity, field_name in ECONOMIC_QUANTITY_FIELDS.items():
        value = record.get(field_name)
        if value is None:
            continue

        # GOR (scf/stb) and water-gas ratio (stb/scf) both carry a gas
        # term. The Mscf/scf deck convention applies to that term the
        # same way it does to a standalone gas rate.
        if quantity == EconomicQuantity.GOR:
            value = from_deck_gas_rate(value, unit_system)
        elif quantity is EconomicQuantity.WATER_GAS_RATIO and unit_system is UnitSystem.FIELD:
            value /= c.MSCF_TO_SCF
        limits.append(
            EconomicLimit(
                quantity=quantity,
                max_value=value,
                workover_action=workover_action,
                end_run=end_run,
                unit_system=unit_system,
            )
        )
    return tuple(limits)


def apply_economic_limits(
    controls: WellControls,
    wecon_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    current_time: float = 0.0,
) -> None:
    """
    Adds each well's economic limits onto its existing control in
    `controls`, in place, using whichever `WECON` record is in effect for
    that well at a given point in the schedule.

    If a well has more than one `WECON` record over the schedule, only the
    most recent one that has already taken effect is used. A reissue
    replaces that well's earlier economic limits rather than adding to
    them, matching how a well's other controls are reissued.

    :param controls: Well controls to update.
    :param wecon_records: Every `WECON` record in the deck.
    :param unit_system: The deck's unit system.
    :param current_time: The point on the schedule clock to resolve limits
        for, in the deck's time unit. Defaults to zero, the start of the run.
    :raises KeyError: If a record's well has no control set in `controls` yet.
    """
    current_records = select_current_records(wecon_records, key="well", current_time=current_time)
    for well_name, record in current_records.items():
        current_control = controls[well_name]
        new_limits = load_economic_limits_from_record(record, unit_system=unit_system)
        if not new_limits:
            continue
        new_quantities = {limit.quantity for limit in new_limits}
        kept_limits = tuple(
            limit
            for limit in current_control.limits
            if not (isinstance(limit, EconomicLimit) and limit.quantity in new_quantities)
        )
        controls.set(well_name, attrs.evolve(current_control, limits=kept_limits + new_limits))


def apply_weltarg(control: WellControl, record: typing.Mapping[str, typing.Any]) -> WellControl:
    """
    Applies one `WELTARG` record to an already-resolved control, changing
    only its mode and the one target value the record names, leaving
    everything else (limits, efficiency factor, guide rate) as it was.

    :param control: The well's control just before this `WELTARG`.
    :param record: One parsed `WELTARG` record.
    :returns: `control` with its mode and matching target field updated.
    :raises ValidationError: If the record's `control_mode` doesn't apply
        to `control`'s kind (e.g. an oil-rate target on an injector).
    """
    control_mode = record["control_mode"]
    try:
        new_mode = (
            PRODUCER_CONTROL_MODE_MAP[control_mode]
            if isinstance(control, ProducerControl)
            else INJECTOR_CONTROL_MODE_MAP[control_mode]
        )
    except KeyError:
        raise ValidationError(
            f"`WELTARG` control mode {control_mode!r} doesn't apply to a "
            f"{'producer' if isinstance(control, ProducerControl) else 'injector'}."
        ) from None

    target_field = WELTARG_TARGET_FIELD[control_mode]
    if target_field is None:
        return attrs.evolve(control, mode=new_mode)
    return attrs.evolve(control, mode=new_mode, **{target_field: record["value"]})


def load_controls_from_records(
    wconprod_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    wconinje_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    weltarg_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    current_time: float = 0.0,
) -> WellControls:
    """
    Builds well controls from every `WCONPROD`, `WCONINJE`, and `WELTARG`
    record in a deck, replayed in schedule order up to a given point in time.

    A well with more than one control record over the course of the
    schedule is changing control mode partway through the run, or in some
    cases converting between producer and injector. A `WCONPROD`/
    `WCONINJE` record replaces the well's control outright; a `WELTARG`
    record only changes the one target value (and mode) it names, leaving
    everything else about the well's current control as it was.

    :param wconprod_records: Every `WCONPROD` record in the deck.
    :param wconinje_records: Every `WCONINJE` record in the deck.
    :param unit_system: The deck's unit system.
    :param weltarg_records: Every `WELTARG` record in the deck, if any.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Well controls keyed by well name, one per well that has a
        control in effect by this time.
    :raises ValidationError: If a `WELTARG` record's well has no
        `WCONPROD`/`WCONINJE` control yet at the point it takes effect.
    """
    # (schedule_time, priority, kind, record). priority orders a same-time
    # `WCONPROD`/`WCONINJE` before a `WELTARG`, so a full control restatement
    # establishes the baseline a same-timestep `WELTARG` then fine-tunes,
    # rather than the two racing in file order.
    events_by_well: dict[str, list[tuple[float, int, str, typing.Mapping[str, typing.Any]]]] = {}
    for record in wconprod_records:
        events_by_well.setdefault(record["well"], []).append((
            record.get("schedule_time", 0.0),
            0,
            "producer",
            record,
        ))
    for record in wconinje_records:
        events_by_well.setdefault(record["well"], []).append((
            record.get("schedule_time", 0.0),
            0,
            "injector",
            record,
        ))
    if weltarg_records:
        for record in weltarg_records:
            events_by_well.setdefault(record["well"], []).append((
                record.get("schedule_time", 0.0),
                1,
                "target",
                record,
            ))

    controls: dict[str, WellControl] = {}
    for well_name, events in events_by_well.items():
        events.sort(key=lambda event: (event[0], event[1]))
        current: WellControl | None = None
        for schedule_time, _, kind, record in events:
            if schedule_time > current_time:
                continue
            if kind == "producer":
                current = load_producer_control_from_record(record, unit_system=unit_system)
            elif kind == "injector":
                current = load_injector_control_from_record(record, unit_system=unit_system)
            else:
                if current is None:
                    raise ValidationError(
                        f"`WELTARG` references well {well_name!r} before it has any "
                        "`WCONPROD`/`WCONINJE` control to modify."
                    )
                current = apply_weltarg(current, record)
        if current is not None:
            controls[well_name] = current
    return WellControls(controls=controls)


def load_groups_from_records(
    gruptree_records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> WellGroups:
    """
    Build a `WellGroups` hierarchy from `GRUPTREE` records.

    :param gruptree_records: All parsed `GRUPTREE` records (`child`/`parent` fields).
    :returns: `WellGroups`.
    """
    groups = {
        record["child"]: WellGroup(name=record["child"], parent=record["parent"])
        for record in gruptree_records
    }
    return WellGroups(groups=groups)


def load_group_control_from_record(
    record: typing.Mapping[str, typing.Any],
    *,
    is_injection: bool,
    unit_system: UnitSystem,
) -> GroupControl:
    """
    Build a `GroupControl` from one `GCONPROD`/`GCONINJE` record.

    :param record: One parsed `GCONPROD` or `GCONINJE` record.
    :param is_injection: `True` for a `GCONINJE` record, `False` for `GCONPROD`.
    :param unit_system: The deck's unit system.
    :returns: Constructed `GroupControl`.
    """
    if is_injection:
        return GroupControl(
            mode=GROUP_INJECTOR_CONTROL_MODE_MAP[record["control_mode"]],
            target_rate=record.get("rate"),
            injected_phase=(
                FluidPhase(record["injector_type"].lower())
                if record.get("injector_type")
                else None
            ),
            unit_system=unit_system,
        )

    control_mode = record["control_mode"]
    if control_mode == "ORAT":
        target_rate = record.get("oil_rate")
    elif control_mode == "WRAT":
        target_rate = record.get("water_rate")
    elif control_mode == "GRAT":
        target_rate = record.get("gas_rate")
    elif control_mode == "LRAT":
        target_rate = record.get("liquid_rate")
    elif control_mode == "RESV":
        target_rate = record.get("reservoir_volume_rate")
    else:
        target_rate = None

    return GroupControl(
        mode=GROUP_PRODUCER_CONTROL_MODE_MAP[control_mode],
        target_rate=target_rate,
        unit_system=unit_system,
    )


def load_group_controls_from_records(
    gconprod_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    gconinje_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    current_time: float = 0.0,
) -> GroupControls:
    """
    Builds group controls from every `GCONPROD` and `GCONINJE` record in a
    deck, resolved to whichever control is actually in effect for each
    group at a given point in the schedule.

    A group with more than one control record over the course of the
    schedule is changing control mode partway through the run. Only the
    most recent one that has already taken effect is used, comparing
    `GCONPROD` and `GCONINJE` records for the same group against each other by
    their actual time in the schedule.

    :param gconprod_records: Every `GCONPROD` record in the deck.
    :param gconinje_records: Every `GCONINJE` record in the deck.
    :param unit_system: The deck's unit system.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Group controls keyed by group name, one per group that has a
        control in effect by this time.
    """
    candidates: list[tuple[typing.Mapping[str, typing.Any], bool]] = [
        (record, False) for record in gconprod_records
    ] + [(record, True) for record in gconinje_records]

    current: dict[str, tuple[typing.Mapping[str, typing.Any], bool]] = {}
    current_times: dict[str, float] = {}
    for record, is_injection in candidates:
        schedule_time = record.get("schedule_time", 0.0)
        if schedule_time > current_time:
            continue
        group_name = record["group"]
        if group_name not in current or schedule_time >= current_times[group_name]:
            current[group_name] = (record, is_injection)
            current_times[group_name] = schedule_time

    controls: dict[str, GroupControl] = {}
    for group_name, (record, is_injection) in current.items():
        controls[group_name] = load_group_control_from_record(
            record, is_injection=is_injection, unit_system=unit_system
        )
    return GroupControls(controls=controls)


def apply_guide_rates(
    controls: WellControls,
    wgrupcon_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    current_time: float = 0.0,
) -> None:
    """
    Sets each well's guide rate on its existing control in `controls`, in
    place, using whichever `WGRUPCON` record is in effect for that well at a
    given point in the schedule.

    :param controls: Well controls to update.
    :param wgrupcon_records: Every `WGRUPCON` record in the deck.
    :param current_time: The point on the schedule clock to resolve guide
        rates for, in the deck's time unit. Defaults to zero, the start of the run.
    :raises KeyError: If a record's well has no control set in `controls` yet.
    """
    current_records = select_current_records(
        wgrupcon_records, key="well", current_time=current_time
    )
    for well_name, record in current_records.items():
        guide_rate = record.get("guide_rate")
        if guide_rate is None:
            continue
        current_control = controls[well_name]
        controls.set(well_name, attrs.evolve(current_control, guide_rate=guide_rate))


def apply_d_factors(
    wells: Wells,
    wdfac_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    current_time: float = 0.0,
) -> None:
    """
    Sets each well's non-Darcy coefficient on its existing `Well` in
    `wells`, in place, using whichever `WDFAC` record is in effect for
    that well at a given point in the schedule.

    :param wells: Wells to update.
    :param wdfac_records: Every `WDFAC` record in the deck.
    :param current_time: The point on the schedule clock to resolve
        `d_factor` for, in the deck's time unit. Defaults to zero, the
        start of the run.
    :raises KeyError: If a record's well isn't in `wells` yet.
    """
    current_records = select_current_records(wdfac_records, key="well", current_time=current_time)
    for well_name, record in current_records.items():
        d_factor = record.get("d_factor")
        if d_factor is None:
            continue
        wells.wells[well_name] = attrs.evolve(wells[well_name], d_factor=d_factor)


def load_wells(deck_file: DeckFile, grid: Grid, current_time: float = 0.0) -> Wells:
    """
    Builds the full well roster from a parsed deck, covering every well
    and completion the deck ever defines across the whole schedule.

    :param deck_file: Parsed deck containing `WELSPECS`, `COMPDAT`, and `WCONINJE`.
    :param grid: Grid built from the same deck, used to resolve completion depths.
    :param current_time: The point on the schedule clock the roster is
        being built for, in the deck's time unit. Anything scheduled at or
        before this time is marked active; anything later is marked
        pending. Defaults to zero, the start of the run.
    :returns: Wells for every well the deck defines, at any point in the schedule.
    :raises ValidationError: If the deck has no grid dimensions.
    """
    if deck_file.dimensions is None:
        raise ValidationError(
            "Deck has no `SPECGRID`/`DIMENS`; `COMPDAT` (I, J, K) can't be resolved."
        )

    welspecs = deck_file.get("WELSPECS") or []
    compdat = deck_file.get("COMPDAT") or []
    wconinje = deck_file.get("WCONINJE") or []
    wpimult = deck_file.get("WPIMULT")
    welopen = deck_file.get("WELOPEN")
    injector_names = {record["well"] for record in wconinje}
    return load_wells_from_records(
        grid=grid,
        welspecs_records=welspecs,
        compdat_records=compdat,
        wpimult_records=wpimult,
        welopen_records=welopen,
        unit_system=deck_file.unit_system,
        injector_names=injector_names,
        current_time=current_time,
    )


def load_well_controls(deck_file: DeckFile, current_time: float = 0.0) -> WellControls:
    """
    Builds well controls from a parsed deck, resolved to whatever is
    actually in effect for each well at a given point in the schedule.

    :param deck_file: Parsed deck containing `WCONPROD`, `WCONINJE`,
        `WELTARG`, `WECON`, and `WGRUPCON`.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Well controls for every well that has a control in effect by this time.
    """
    controls = load_controls_from_records(
        wconprod_records=deck_file.get("WCONPROD") or [],
        wconinje_records=deck_file.get("WCONINJE") or [],
        weltarg_records=deck_file.get("WELTARG"),
        unit_system=deck_file.unit_system,
        current_time=current_time,
    )
    wecon = deck_file.get("WECON") or []
    if wecon:
        apply_economic_limits(
            controls=controls,
            wecon_records=wecon,
            unit_system=deck_file.unit_system,
            current_time=current_time,
        )

    wgrupcon = deck_file.get("WGRUPCON") or []
    if wgrupcon:
        apply_guide_rates(controls=controls, wgrupcon_records=wgrupcon, current_time=current_time)
    return controls


def load_groups(deck_file: DeckFile) -> WellGroups:
    """
    Builds the group hierarchy from a parsed deck.

    :param deck_file: Parsed deck.
    :returns: Well groups from `GRUPTREE`.
    :raises DeckParseError: If the deck has no `GRUPTREE`.
    """
    gruptree = deck_file.get("GRUPTREE")
    if not gruptree:
        raise DeckParseError("Cannot load well groups from deck. `GRUPTREE` is missing.")
    return load_groups_from_records(gruptree)


def load_group_controls(deck_file: DeckFile, current_time: float = 0.0) -> GroupControls:
    """
    Builds group controls from a parsed deck, resolved to whatever is
    actually in effect for each group at a given point in the schedule.

    :param deck_file: Parsed deck.
    :param current_time: The point on the schedule clock to resolve
        controls for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :returns: Group controls for every group that has a control in effect by this time.
    :raises DeckParseError: If the deck has neither `GCONPROD` nor `GCONINJE`.
    """
    gconprod = deck_file.get("GCONPROD") or []
    gconinje = deck_file.get("GCONINJE") or []
    if not gconprod and not gconinje:
        raise DeckParseError(
            "Cannot load well group controls from deck. `GCONPROD` and `GCONINJE` are both missing. "
            "At least one should be present."
        )
    return load_group_controls_from_records(
        gconprod_records=gconprod,
        gconinje_records=gconinje,
        unit_system=deck_file.unit_system,
        current_time=current_time,
    )


def load_schedule(
    deck_file: DeckFile, *, compiled_at: float = 0.0
) -> Schedule["CompiledBlackOilModel"]:
    """
    Builds a `Schedule[CompiledBlackOilModel]` from every well-editing
    keyword in a deck's schedule section, for events strictly after
    `compiled_at`.

    Covers every well ever mentioned anywhere in the deck, not just ones
    active at `compiled_at`. `load_wells`/`compile_well_system` already
    compiles the whole roster up front, tagging a well or completion not
    yet due as `PENDING` rather than leaving it out (see
    `compile_well_system`'s own docstring). A `WELSPECS` after
    `compiled_at` becomes an `ActivateWell` action, and a `COMPDAT` after
    `compiled_at` becomes an `ActivateCompletion` action. Both are just
    status flips against rows that already exist, and not a new-row
    allocation. Nothing here reloads or recompiles anything.

    Every `WECON` record with `schedule_time <= compiled_at` is assumed
    already baked into the model's economic limits at compile time (see
    `apply_economic_limits`), and is skipped. A `WECON` reissue after
    `compiled_at` becomes one `SetLimit` action per quantity the record
    defines (a single record can set several at once, a max water cut
    and a max GOR together, say).

    A `WCONPROD`/`WCONINJE` reissue sets the well's control mode and
    targets. When its own `bhp`/`thp` item is given and the mode isn't
    that item, it also patches the well's implicit `BHPLimit`/`THPLimit`
    via an additional `SetLimit` action.

    Every `SetLimit` action here can only patch a limit row the well
    already had for that kind/quantity at compile time. The
    `CompiledLimits` CSR table can't grow a new row mid-schedule.
    `RateLimit` has no deck-record source in this codebase at all, so
    it's never emitted here.

    :param deck_file: The deck to read schedule keywords from.
    :param compiled_at: The point on the schedule clock the model this
        schedule will run against was compiled at, in the deck's time
        unit. Only events strictly after this become actions.
    :returns: A `Schedule[CompiledBlackOilModel]`, one rule per
        qualifying record, sorted by `schedule_time`.
    """
    unit_system = deck_file.unit_system
    rules: list[Rule[CompiledBlackOilModel]] = []
    empty = []

    def is_due(record: typing.Mapping[str, typing.Any]) -> bool:
        return record.get("schedule_time", 0.0) > compiled_at

    for record in deck_file.get("WELSPECS") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = ActivateWell(well_name=record["well"])
        rules.append(
            Rule(
                event=TimeEvent(at=schedule_time),
                action=action,
                name=f"welspecs:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WELOPEN") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = OpenWell(
            well_name=record["well"],
            status=WELOPEN_STATUS_MAP[record["status"]],
            i=record.get("i", 0),
            j=record.get("j", 0),
            k1=record.get("k1", 0),
            k2=record.get("k2", 0),
        )
        rules.append(
            Rule(
                event=TimeEvent(at=schedule_time),
                action=action,
                name=f"welopen:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WPIMULT") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = MultiplyConnectionFactor(
            well_name=record["well"],
            multiplier=record["multiplier"],
            i=record.get("i", 0),
            j=record.get("j", 0),
            k1=record.get("k1", 0),
            k2=record.get("k2", 0),
        )
        rules.append(
            Rule(
                event=TimeEvent(at=schedule_time),
                action=action,
                name=f"wpimult:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WELTARG") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = SetWellTarget(
            well_name=record["well"],
            control_mode=record["control_mode"],
            value=record.get("value"),
        )
        rules.append(
            Rule(
                event=TimeEvent(at=schedule_time),
                action=action,
                name=f"weltarg:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WCONPROD") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        control = load_producer_control_from_record(record, unit_system=unit_system)
        action = SetWellControl(
            well_name=record["well"],
            mode=control.mode,
            target_rate=control.target_rate,
            target_bhp=control.target_bhp,
            target_thp=control.target_thp,
        )
        rules.append(
            Rule(
                event=TimeEvent(at=schedule_time),
                action=action,
                name=f"wconprod:{record['well']}@{schedule_time}",
            )
        )
        for limit in control.limits:
            if isinstance(limit, BHPLimit):
                kind = LimitKind.BHP
            elif isinstance(limit, THPLimit):
                kind = LimitKind.THP
            else:
                continue
            limit_action = SetLimit(
                well_name=record["well"],
                kind=kind,
                min_value=limit.min_value,
                max_value=limit.max_value,
            )
            rules.append(
                Rule(
                    event=TimeEvent(at=schedule_time),
                    action=limit_action,
                    name=f"wconprod-{kind.name.lower()}limit:{record['well']}@{schedule_time}",
                )
            )

    for record in deck_file.get("WCONINJE") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        control = load_injector_control_from_record(record, unit_system=unit_system)
        action = SetWellControl(
            well_name=record["well"],
            mode=control.mode,
            target_rate=control.target_rate,
            target_bhp=control.target_bhp,
            target_thp=control.target_thp,
            injected_phase=control.injected_phase,
        )
        rules.append(
            Rule(
                event=TimeEvent(at=schedule_time),
                action=action,
                name=f"wconinje:{record['well']}@{schedule_time}",
            )
        )
        for limit in control.limits:
            if isinstance(limit, BHPLimit):
                kind = LimitKind.BHP
            elif isinstance(limit, THPLimit):
                kind = LimitKind.THP
            else:
                continue
            limit_action = SetLimit(
                well_name=record["well"],
                kind=kind,
                min_value=limit.min_value,
                max_value=limit.max_value,
            )
            rules.append(
                Rule(
                    event=TimeEvent(at=schedule_time),
                    action=limit_action,
                    name=f"wconinje-{kind.name.lower()}limit:{record['well']}@{schedule_time}",
                )
            )

    for record in deck_file.get("WECON") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        for limit in load_economic_limits_from_record(record, unit_system=unit_system):
            action = SetLimit(
                well_name=record["well"],
                kind=LimitKind.ECONOMIC,
                quantity=limit.quantity,
                min_value=limit.min_value,
                max_value=limit.max_value,
                workover_action=limit.workover_action,
                end_run=limit.end_run,
            )
            rules.append(
                Rule(
                    event=TimeEvent(at=schedule_time),
                    action=action,
                    name=f"wecon:{record['well']}:{limit.quantity}@{schedule_time}",
                )
            )

    for record in deck_file.get("COMPDAT") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = ActivateCompletion(
            well_name=record["well"],
            i=record["i"],
            j=record["j"],
            k1=record["k1"],
            k2=record["k2"],
        )
        rules.append(
            Rule(
                event=TimeEvent(at=schedule_time),
                action=action,
                name=f"compdat:{record['well']}@{schedule_time}",
            )
        )

    rules.sort(key=lambda rule: rule.event.at)  # type: ignore[attr-defined]
    return Schedule(rules=tuple(rules))
