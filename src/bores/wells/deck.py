"""
Utilities for building well model definition objects from parsed Eclipse deck records.
"""

import itertools
import typing

import attrs
import numpy as np

from bores.constants import c
from bores.deck.core import DeckParseError
from bores.deck.file import DeckFile
from bores.errors import NotSupportedError, ValidationError
from bores.grids.base import Grid
from bores.schedule.actions import RunSequence
from bores.schedule.base import Action, Event, Schedule, ScheduleItem
from bores.schedule.events import AllOf, AnyOf, ComparisonOperator, TimeEvent, TimeWindowEvent
from bores.schedule.summary import RecordSummary, Summary
from bores.timestep.targets import TimeTargets
from bores.types import FluidPhase, Number, UnitSystem
from bores.units import get_conversion_factors
from bores.wells import summary as ws
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
    RateQuantity,
    THPLimit,
    WellControl,
    WellControls,
    WellTargetMode,
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
    WELSPECS_PHASE_MAP,
    WELTARG_TARGET_FIELD,
)
from bores.wells.schedule import (
    ActivateCompletions,
    ActivateWells,
    GasOilRatioThreshold,
    MultiplyConnectionFactors,
    RateThreshold,
    SetGroupLimit,
    SetLimits,
    SetWellControls,
    SetWellLift,
    SetWellTargets,
    UpdateWellStatuses,
    WaterCutThreshold,
    match_connection_target,
)
from bores.wells.trajectory import TrajectoryStation, WellTrajectory

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
    "apply_well_segments",
    "from_deck_gas_rate",
    "get_fluid_in_place_regions",
    "is_whole_well_record",
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
    "load_summary_items",
    "load_well_controls",
    "load_well_from_records",
    "load_well_segments",
    "load_wells",
    "load_wells_from_records",
    "replay_open_status",
    "select_current_records",
]


def connection_layers(record: typing.Mapping[str, typing.Any], *, keyword: str) -> tuple[int, int]:
    """
    The first and last k layer a `WELOPEN` or `WPIMULT` record targets, with `0` meaning any.

    A record parsed from a deck carries one `k`, so both layers are that value. A record
    built in code may give a layer range as `k1` and `k2` instead.

    :param record: `WELOPEN` or `WPIMULT` record.
    :param keyword: Name of the keyword the record came from, for error messages.
    :returns: The first and last layer.
    :raises NotSupportedError: If the record selects connections by completion number
        (`c1`/`c2`), which is not supported.
    """
    if record.get("c1", 0) or record.get("c2", 0):
        raise NotSupportedError(
            f"`{keyword}` for well {record['well']!r} selects connections by completion "
            "number (items 6 and 7), which is not supported. Select them by i, j and k."
        )
    if "k" not in record and ("k1" in record or "k2" in record):
        return record.get("k1", 0), record.get("k2", 0)
    layer = record.get("k", 0)
    return layer, layer


def is_whole_well_record(record: typing.Mapping[str, typing.Any], *, keyword: str) -> bool:
    """
    :param record: A `WELOPEN` or `WPIMULT` record.
    :param keyword: Name of the keyword the record came from, for error messages.
    :returns: Whether the record names no connection, so it applies to the well as a whole.
    :raises NotSupportedError: If the record selects connections by completion number.
    """
    first_layer, last_layer = connection_layers(record, keyword=keyword)
    return not (record.get("i", 0) or record.get("j", 0) or first_layer or last_layer)


def replay_open_status(
    *,
    control_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None,
    welopen_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None,
    current_time: float,
) -> CompletionStatus:
    """
    A well's own open or shut flag at a point in the schedule, found by replaying every record
    that sets it, in schedule order.

    A `WCONPROD` or `WCONINJE` record sets it through its status item, and an omitted status
    means `OPEN`. A `WELOPEN` record sets it when it names no connection. At the same time,
    the `WELOPEN` is replayed last, so it has the last word, as it does in `load_schedule`.
    A well that nothing sets is open.

    :param control_records: The well's `WCONPROD` and `WCONINJE` records, from any point in
        the schedule.
    :param welopen_records: The well's `WELOPEN` records, from any point in the schedule.
    :param current_time: Records scheduled after this time are not replayed yet.
    :returns: The well's open or shut status as of `current_time`.
    """
    changes: list[tuple[float, int, CompletionStatus]] = []
    for record in control_records or ():
        time = record.get("schedule_time", 0.0)
        if time <= current_time:
            changes.append((
                time,
                0,
                WELOPEN_STATUS_MAP[str(record.get("status") or "OPEN").upper()],
            ))
    for record in welopen_records or ():
        time = record.get("schedule_time", 0.0)
        if time <= current_time and is_whole_well_record(record, keyword="WELOPEN"):
            changes.append((time, 1, WELOPEN_STATUS_MAP[record["status"]]))
    status = CompletionStatus.OPEN
    for *_, new_status in sorted(changes, key=lambda change: change[:2]):
        status = new_status
    return status


def load_well_from_records(
    grid: Grid,
    welspecs_record: typing.Mapping[str, typing.Any],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    welopen_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    control_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
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

    Any `WELOPEN` record for the well that names connections, up to `current_time`, is
    replayed in schedule order afterwards, so a completion opened or shut later in the
    schedule without a fresh `COMPDAT` still ends up with the right status. The well's own
    open or shut flag is found separately (`replay_open_status`), from the `WELOPEN` records
    that name no connection and the status item of its `WCONPROD`/`WCONINJE` records. It does
    not change the status of any connection.

    :param grid: Grid the well's completions are resolved against.
    :param welspecs_record: The well's `WELSPECS` record.
    :param compdat_records: Every `COMPDAT` record for this well, from any
        point in the schedule.
    :param wpimult_records: Every `WPIMULT` record for this well, if any.
    :param welopen_records: Every `WELOPEN` record for this well, if any.
    :param control_records: Every `WCONPROD` or `WCONINJE` record for this well, if any. Only
        their status item is used here, for the well's open or shut flag. The controls
        themselves are loaded by `load_controls_from_records`.
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

    # Records are replayed in schedule order, so a connection re-entered later is a later
    # version of the same connection.
    compdat_records = sorted(compdat_records, key=lambda r: r.get("schedule_time", 0.0))

    # A connection outside the grid is rejected here. Plain index arithmetic would put an `i`
    # past the last column onto the next row, and a `j` past the last row onto the next layer.
    connection_i = np.array([record["i"] for record in compdat_records])
    connection_j = np.array([record["j"] for record in compdat_records])
    connection_k1 = np.array([record["k1"] for record in compdat_records])
    connection_k2 = np.array([record["k2"] for record in compdat_records])
    outside = ~(
        (connection_i >= 1)
        & (connection_i <= dims.nx)
        & (connection_j >= 1)
        & (connection_j <= dims.ny)
        & (connection_k1 >= 1)
        & (connection_k2 <= dims.nz)
        & (connection_k1 <= connection_k2)
    )
    if outside.any():
        bad = int(np.flatnonzero(outside)[0])
        raise ValidationError(
            f"Well {welspecs_record['well']!r} has a `COMPDAT` connection at "
            f"i={connection_i[bad]}, j={connection_j[bad]}, k={connection_k1[bad]}-"
            f"{connection_k2[bad]}, which is not within the {dims.nx} x {dims.ny} x {dims.nz} "
            "grid (indices start at 1 and `k1` must not exceed `k2`)."
        )
    # Minus 1, to move from 1-based to 0-based indexing used internally
    top_cells = dims.flat_index(connection_i - 1, connection_j - 1, connection_k1 - 1)
    bottom_cells = dims.flat_index(connection_i - 1, connection_j - 1, connection_k2 - 1)

    for position, record in enumerate(compdat_records):
        i, j = record["i"], record["j"]
        k1, k2 = record["k1"], record["k2"]
        top_cell = int(top_cells[position])
        bottom_cell = int(bottom_cells[position])
        top_depth = grid.cell_min_xyz[top_cell, 2]
        bottom_depth = grid.cell_max_xyz[bottom_cell, 2]
        direction = record.get("direction")
        saturation_region = record.get("saturation_table") or None  # 0 should map to None too
        radius = (record.get("diameter") or 0) * 0.5
        skin = record.get("skin", 0.0) or 0.0
        status = (
            CompletionStatus.OPEN
            if record.get("status", "OPEN") == "OPEN"
            else CompletionStatus.SHUT
        )
        perforation_specs.append({
            "i": i,
            "j": j,
            "k1": k1,
            "k2": k2,
            "top_depth": top_depth,
            "bottom_depth": bottom_depth,
            "cell_index": top_cell if k1 == k2 else None,
            "skin": skin,
            "wellbore_radius": radius,
            "status": status,
            "saturation_region": saturation_region,
            "connection_factor_override": record.get("connection_factor"),
            "connection_factor_multiplier": None,
            "direction": DIRECTION_MAP.get(direction) if direction else None,
            "start_time": record.get("schedule_time", 0.0),
            "end_time": float("inf"),
        })

    # A `COMPDAT` for a connection the well already has replaces it, so each connection can
    # have several versions, each in effect from its own start until the next one starts.
    # Overlapping layer ranges that are not identical would need an existing completion split
    # in two, which is not supported.
    versions: dict[tuple[int, int, int, int], list[dict[str, typing.Any]]] = {}
    ranges_by_column: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for spec in perforation_specs:
        versions.setdefault((spec["i"], spec["j"], spec["k1"], spec["k2"]), []).append(spec)
        ranges_by_column.setdefault((spec["i"], spec["j"]), set()).add((spec["k1"], spec["k2"]))

    for (column_i, column_j), ranges in ranges_by_column.items():
        for first, second in itertools.combinations(sorted(ranges), 2):
            if first[0] <= second[1] and second[0] <= first[1]:
                raise NotSupportedError(
                    f"Well {welspecs_record['well']!r} has `COMPDAT` records at i={column_i}, "
                    f"j={column_j} for the layers {first} and {second}, which overlap without "
                    "being the same range. Re-enter a connection with its original layer range."
                )
    for group in versions.values():
        for earlier, later in itertools.pairwise(group):
            earlier["end_time"] = later["start_time"]

    def in_effect(spec: typing.Mapping[str, typing.Any], time: float) -> bool:
        return spec["start_time"] <= time < spec["end_time"]

    def select(
        record: typing.Mapping[str, typing.Any], *, keyword: str
    ) -> list[dict[str, typing.Any]]:
        """
        The specs a `WPIMULT` or `WELOPEN` record acts on, as of its own time.

        Raises `ValidationError` if the record matches none.
        """
        time = record.get("schedule_time", 0.0)
        i, j = record.get("i", 0), record.get("j", 0)
        k1, k2 = connection_layers(record, keyword=keyword)
        matched = [
            spec
            for spec in perforation_specs
            if in_effect(spec, time)
            and match_connection_target(
                i=i,
                j=j,
                k1=k1,
                k2=k2,
                connection_i=spec["i"],
                connection_j=spec["j"],
                connection_k1=spec["k1"],
                connection_k2=spec["k2"],
            )
        ]
        if not matched:
            raise ValidationError(
                f"`{keyword}` for well {welspecs_record['well']!r} at i={i}, j={j}, "
                f"k1={k1}, k2={k2} matches no connection of the well in effect at that time."
            )
        return matched

    if wpimult_records:
        # Every `WPIMULT` up to `current_time` scales the connections in effect at its time, and
        # they accumulate: two of 0.5 on one connection leave it at 0.25. A connection that has
        # not started yet is not scaled, and re-entering a connection starts it from a fresh
        # factor. Later records are left to the schedule, which applies them at their own time.
        for record in sorted(
            (r for r in wpimult_records if r.get("schedule_time", 0.0) <= current_time),
            key=lambda r: r.get("schedule_time", 0.0),
        ):
            for spec in select(record, keyword="WPIMULT"):
                spec["connection_factor_multiplier"] = (
                    spec["connection_factor_multiplier"] or 1.0
                ) * record["multiplier"]

    if welopen_records:
        # Open/shut events that name connections, replayed in schedule order up to
        # `current_time`. Each only touches connections in effect at its time whose own
        # (i, j) column and k-layer range contain the event's k layer. An event that names
        # no connection is the well's own flag, which is found by `replay_open_status`.
        for record in sorted(
            (
                r
                for r in welopen_records
                if r.get("schedule_time", 0.0) <= current_time
                and not is_whole_well_record(r, keyword="WELOPEN")
            ),
            key=lambda r: r.get("schedule_time", 0.0),
        ):
            new_status = WELOPEN_STATUS_MAP[record["status"]]
            for spec in select(record, keyword="WELOPEN"):
                spec["status"] = new_status

    # Versions already replaced by `current_time` are history, and are left out.
    perforation_specs = [spec for spec in perforation_specs if spec["end_time"] > current_time]
    for spec in perforation_specs:
        spec["schedule_status"] = (
            WellStatus.ACTIVE if spec["start_time"] <= current_time else WellStatus.PENDING
        )

    perforations = [
        Perforation(
            top_depth=spec["top_depth"],
            bottom_depth=spec["bottom_depth"],
            cell_index=spec["cell_index"],
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
    deepest_bottom_depth = max(
        perforation.bottom_depth
        for perforation in perforations
        if perforation.bottom_depth is not None
    )
    head_i, head_j = welspecs_record["i"], welspecs_record["j"]
    if not (1 <= head_i <= dims.nx and 1 <= head_j <= dims.ny):
        raise ValidationError(
            f"Well {welspecs_record['well']!r} has its `WELSPECS` head at i={head_i}, j={head_j}, "
            f"which is not within the {dims.nx} x {dims.ny} grid (indices start at 1)."
        )

    surface_location = grid.get_cell_center_at(
        head_i - 1,
        head_j - 1,
        0,  # At surface
    )[:2]
    pvt_region = welspecs_record.get("pvt_table") or None  # 0 should map to None too
    preferred_phase = (
        WELSPECS_PHASE_MAP[welspecs_record["phase"]] if welspecs_record.get("phase") else None
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
        open_status=replay_open_status(
            control_records=control_records,
            welopen_records=welopen_records,
            current_time=current_time,
        ),
        schedule_status=well_schedule_status,
    )


def load_well_segments(
    deck_file: DeckFile,
    wells: Wells,
    *,
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
) -> Wells:
    """
    Reads every `WELSEGS`/`COMPSEGS` record in `deck_file` and layers
    measured depth onto `wells`' already-built `COMPDAT` perforations.

    A well's trajectory is static, load-once geometry, the same as its
    perforations. Every `WELSEGS`/`COMPSEGS` record is applied regardless
    of its `schedule_time`, matching how `load_wells` already includes
    every `COMPDAT` record for a well no matter when it takes effect.

    :param deck_file: The deck to read from.
    :param wells: An already-built `Wells`, from `load_wells`.
    :param compdat_records: Every `COMPDAT` record in the deck, in the
        same order `wells` was built from. Defaults to `deck_file.get("COMPDAT")`.
    :returns: `wells`, with every well named in a `WELSEGS` record
        updated; a well `WELSEGS` doesn't mention is returned unchanged.
    """
    welsegs_records = deck_file.get("WELSEGS")
    compsegs_records = deck_file.get("COMPSEGS")
    if not welsegs_records or not compsegs_records:
        return wells
    resolved_compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]] = (
        compdat_records if compdat_records is not None else (deck_file.get("COMPDAT") or [])
    )
    return apply_well_segments(
        wells,
        welsegs_records=welsegs_records,
        compsegs_records=compsegs_records,
        compdat_records=resolved_compdat_records,
    )


def apply_well_segments(
    wells: Wells,
    *,
    welsegs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compsegs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> Wells:
    """
    Layers measured depth onto a well's already-built `COMPDAT`
    perforations, from a `WELSEGS`/`COMPSEGS` pair.

    Only the main bore (`branch == 1`) is supported. A lateral branch
    raises an error rather than being dropped. `COMPDAT` still owns
    which grid cells a well connects to, `WELSEGS`/`COMPSEGS` only add
    each connection's own measured depth on top of that, matching how
    Eclipse itself always requires `COMPDAT` alongside a multi-segment
    well's own `WELSEGS`/`COMPSEGS`, rather than deriving connections
    from a survey the way `bores`'s own `WellTrajectory` does for
    a hand-built deviated well elsewhere.

    Each `COMPSEGS` connection is matched back to a perforation by its
    own `(i, j, k)` against `compdat_records`, not by true vertical
    depth. Several completions on a horizontal or deviated well often
    share the same true vertical depth, so depth alone cannot tell them
    apart. Grid location can htough, since `compdat_records` is what `wells`'
    own perforations were built from, in the same order.

    A well's segment tree gives a measured depth to true vertical depth
    profile with no horizontal position in it (`WELSEGS` carries
    along-hole length and depth change only, not azimuth), so the
    `WellTrajectory` built here has every station's `x`/`y` at `0.0`,
    accurate for true vertical depth and inclination.

    :param wells: An already-built `Wells`, from `load_wells`.
    :param welsegs_records: Every `WELSEGS` occurrence in the deck, from
        `DeckFile.get("WELSEGS")`.
    :param compsegs_records: Every `COMPSEGS` occurrence in the deck, from
        `DeckFile.get("COMPSEGS")`.
    :param compdat_records: Every `COMPDAT` record in the deck, the same
        set `wells` was built from, in the same order.
    :returns: `wells`, with every well named in `welsegs_records` updated;
        a well not named there is returned unchanged.
    :raises NotSupportedError: If a `COMPSEGS` record targets a branch
        other than `1`.
    :raises ValidationError: If a `COMPSEGS` record has no explicit
        `start_length`/`end_length`, names a connection with no matching
        `COMPDAT` perforation, or leaves some of a well's perforations
        without a measured depth.
    """
    connections_by_well: dict[str, list[tuple[int, int, int, int]]] = {}
    for record in compdat_records:
        connections_by_well.setdefault(record["well"], []).append((
            record["i"],
            record["j"],
            record["k1"],
            record["k2"],
        ))

    compsegs_by_well: dict[str, list[typing.Mapping[str, typing.Any]]] = {}
    for occurrence in compsegs_records:
        compsegs_by_well.setdefault(occurrence["well"], []).append(occurrence)

    updated_wells = dict(wells.wells)
    for header in welsegs_records:
        well_name = header["well"]
        occurrences = compsegs_by_well.get(well_name)
        if well_name not in updated_wells or not occurrences:
            continue
        well = updated_wells[well_name]
        connections = connections_by_well.get(well_name, [])
        if len(connections) != len(well.perforations):
            raise ValidationError(
                f"Well {well_name!r} has {len(well.perforations)} perforation(s) but "
                f"{len(connections)} `COMPDAT` connection(s) were found for it. "
                "`compdat_records` must be the same set `wells` was built from, so "
                "`COMPSEGS` connections can be matched back to perforations by position."
            )

        # Segment tree built as a mapping of segment number -> (cumulative measured depth,
        # cumulative true vertical depth), walked from the first segment
        # node (already at (tubing_length_to_first_segment, reference_depth))
        # via each detail record's own outlet_segment. Records are expected
        # in an order where a segment's outlet is already resolved by the
        # time that segment's own record is reached, the same assumption
        # Eclipse itself makes.
        tubing_length_to_first_segment = header["tubing_length_to_first_segment"]
        reference_depth = header["reference_depth"]
        mode = header.get("length_depth_mode") or "INC"
        node_md = {1: tubing_length_to_first_segment}
        node_tvd = {1: reference_depth}
        for detail in header["details"]:
            if detail["branch"] != 1:
                continue  # a lateral's own segments; only the main bore matters here

            outlet = detail["outlet_segment"]
            if outlet not in node_md:
                continue  # outlet not yet resolved; skip rather than guess

            if mode == "ABS":
                md = tubing_length_to_first_segment + detail["length"]
                tvd = reference_depth + detail["depth_change"]
            else:
                md = node_md[outlet] + detail["length"]
                tvd = node_tvd[outlet] + detail["depth_change"]

            for segment in range(detail["first_segment"], detail["last_segment"] + 1):
                node_md[segment] = md
                node_tvd[segment] = tvd

        if len(node_md) <= 1:
            continue  # no main-bore segment resolved; nothing to attach

        points = sorted(set(zip(node_md.values(), node_tvd.values(), strict=True)))
        md_points = [points[0][0]]
        tvd_points = [points[0][1]]
        for md, tvd in points[1:]:
            if md > md_points[-1]:
                md_points.append(md)
                tvd_points.append(tvd)

        new_perforations = list(well.perforations)
        matched_indices: set[int] = set()
        for occurrence in occurrences:
            for completion in occurrence["details"]:
                if completion["branch"] != 1:
                    raise NotSupportedError(
                        f"`COMPSEGS` on well {well_name!r} targets branch "
                        f"{completion['branch']!r}; only the main bore (branch 1) is "
                        "supported by `apply_well_segments` today."
                    )

                start_length = completion.get("start_length")
                end_length = completion.get("end_length")
                if start_length is None or end_length is None:
                    raise ValidationError(
                        f"`COMPSEGS` on well {well_name!r}, connection "
                        f"({completion['i']}, {completion['j']}, {completion['k']}), needs "
                        "an explicit `start_length`/`end_length`; deriving them from the "
                        "connection's own grid-block geometry (deck `1*`) is not supported."
                    )

                i, j, k = completion["i"], completion["j"], completion["k"]
                match_index = next(
                    (
                        index
                        for index, (ci, cj, ck1, ck2) in enumerate(connections)
                        if ci == i and cj == j and ck1 <= k <= ck2
                    ),
                    None,
                )
                if match_index is None:
                    raise ValidationError(
                        f"`COMPSEGS` on well {well_name!r} names connection "
                        f"({i}, {j}, {k}), but no `COMPDAT` connection there was found."
                    )

                top_md = tubing_length_to_first_segment + start_length
                bottom_md = tubing_length_to_first_segment + end_length
                new_perforations[match_index] = attrs.evolve(
                    new_perforations[match_index],
                    top_md=min(top_md, bottom_md),
                    bottom_md=max(top_md, bottom_md),
                    # A completion direction only means something for a TVD-based
                    # perforation resolved via Peaceman's formula; a measured-depth
                    # one always resolves through the isotropic equivalent-radius
                    # well index instead, and `Perforation` itself rejects the two
                    # together (see its own validation).
                    direction=None,
                )
                matched_indices.add(match_index)

        if not matched_indices:
            continue

        unmatched = [
            index for index in range(len(new_perforations)) if index not in matched_indices
        ]
        if unmatched:
            raise ValidationError(
                f"Well {well_name!r} has a `WELSEGS`/`COMPSEGS` trajectory, but "
                f"{len(unmatched)} of its {len(new_perforations)} `COMPDAT` connection(s) "
                f"were never given a measured depth by `COMPSEGS`: "
                f"{[connections[index] for index in unmatched]}. Every connection on a "
                "multi-segment well needs its own `COMPSEGS` entry."
            )

        stations = tuple(
            TrajectoryStation(measured_depth=md, x=0.0, y=0.0, z=tvd)
            for md, tvd in zip(md_points, tvd_points, strict=True)
        )
        updated_wells[well_name] = attrs.evolve(
            well,
            perforations=tuple(new_perforations),
            trajectory=WellTrajectory(stations=stations),
        )

    return Wells(wells=updated_wells, unit_system=wells.unit_system)


def load_wells_from_records(
    grid: Grid,
    welspecs_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    compdat_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    wpimult_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    welopen_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    control_records: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
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
    :param control_records: Every `WCONPROD` and `WCONINJE` record in the deck, if any. Only
        their status item is used here, for each well's open or shut flag.
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
    control_by_well: dict[str, list[typing.Mapping[str, typing.Any]]] = {}
    for record in compdat_records:
        compdat_by_well.setdefault(record["well"], []).append(record)

    if wpimult_records:
        for record in wpimult_records:
            wpimult_by_well.setdefault(record["well"], []).append(record)

    if welopen_records:
        for record in welopen_records:
            welopen_by_well.setdefault(record["well"], []).append(record)

    if control_records:
        for record in control_records:
            control_by_well.setdefault(record["well"], []).append(record)

    wells = {
        record["well"]: load_well_from_records(
            grid,
            welspecs_record=record,
            compdat_records=compdat_by_well.get(record["well"], []),
            wpimult_records=wpimult_by_well.get(record["well"]),
            welopen_records=welopen_by_well.get(record["well"]),
            control_records=control_by_well.get(record["well"]),
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
    :returns: Constructed `ProducerControl`, with the `WCONPROD` VFP table number and ALQ. Adds an
        implicit `BHPLimit(min_value=bhp)`
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

    vfp_table = record.get("vfp_table")
    return ProducerControl(
        mode=mode,
        target_rate=target_rate,
        target_bhp=bhp,
        target_thp=record.get("thp"),
        limits=tuple(limits),
        vfp_table=int(vfp_table) if vfp_table else None,
        artificial_lift_quantity=record.get("alq") or 0.0,
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

    vfp_table = record.get("vfp_table")
    return InjectorControl(
        injected_phase=phase,
        mode=mode,
        target_rate=rate,
        target_bhp=bhp,
        target_thp=record.get("thp"),
        limits=tuple(limits),
        vfp_table=int(vfp_table) if vfp_table else None,
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
    workover_action = WorkoverAction(record.get("workover_action", "NONE"))
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


def load_group_economic_limits_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> tuple[EconomicLimit, ...]:
    """
    Loads a group's economic limits from one `GECON` record.

    Same shape as `load_economic_limits_from_record`, except `GECON`
    carries a fifth quantity `WECON` doesn't, `min_gas_rate`, and its
    workover field is named `workover_procedure`, defaulting to `NONE`
    rather than `WELL`. A group's limit is tracked but not enforced
    unless a procedure is explicitly named.

    :param record: One parsed `GECON` record.
    :param unit_system: The deck's unit system.
    :returns: One economic limit per non-zero threshold present on the
        record (minimum oil rate, minimum gas rate, water cut, GOR,
        water-gas ratio).
    """
    workover_action = WorkoverAction(record.get("workover_procedure", "NONE"))
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


def apply_group_economic_limits(
    group_controls: GroupControls,
    gecon_records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    current_time: float = 0.0,
) -> None:
    """
    Adds each group's economic limits onto its existing control in
    `group_controls`, in place, using whichever `GECON` record is in
    effect for that group at a given point in the schedule.

    A reissue replaces that group's earlier economic limits rather than
    adding to them, matching `apply_economic_limits`' own well-level
    behavior. Deciding which member well to act on when a group's limit
    is breached isn't implemented - this only loads and stores the
    limits themselves.

    :param group_controls: Group controls to update.
    :param gecon_records: Every `GECON` record in the deck.
    :param unit_system: The deck's unit system.
    :param current_time: The point on the schedule clock to resolve
        limits for, in the deck's time unit. Defaults to zero, the start
        of the run.
    :raises KeyError: If a record's group has no control set in `group_controls` yet.
    """
    current_records = select_current_records(gecon_records, key="group", current_time=current_time)
    for group_name, record in current_records.items():
        current_control = group_controls[group_name]
        new_limits = load_group_economic_limits_from_record(record, unit_system=unit_system)
        if not new_limits:
            continue
        new_quantities = {limit.quantity for limit in new_limits}
        kept_limits = tuple(
            limit for limit in current_control.limits if limit.quantity not in new_quantities
        )
        group_controls.set(
            group_name, attrs.evolve(current_control, limits=kept_limits + new_limits)
        )


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
        current_control: WellControl | None = None
        for schedule_time, _, kind, record in events:
            if schedule_time > current_time:
                continue
            if kind == "producer":
                current_control = load_producer_control_from_record(
                    record, unit_system=unit_system
                )
            elif kind == "injector":
                current_control = load_injector_control_from_record(
                    record, unit_system=unit_system
                )
            else:
                if current_control is None:
                    raise ValidationError(
                        f"`WELTARG` references well {well_name!r} before it has any "
                        "`WCONPROD`/`WCONINJE` control to modify."
                    )
                current_control = apply_weltarg(current_control, record)
        if current_control is not None:
            controls[well_name] = current_control
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
        wells[well_name] = attrs.evolve(wells[well_name], d_factor=d_factor)


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
    wconprod = deck_file.get("WCONPROD") or []
    wpimult = deck_file.get("WPIMULT")
    welopen = deck_file.get("WELOPEN")
    injector_names = {record["well"] for record in wconinje}
    wells = load_wells_from_records(
        grid=grid,
        welspecs_records=welspecs,
        compdat_records=compdat,
        wpimult_records=wpimult,
        welopen_records=welopen,
        control_records=[*wconprod, *wconinje],
        unit_system=deck_file.unit_system,
        injector_names=injector_names,
        current_time=current_time,
    )
    return load_well_segments(deck_file, wells, compdat_records=compdat)


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

    :param deck_file: Parsed deck containing `GCONPROD`, `GCONINJE`, and `GECON`.
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
    controls = load_group_controls_from_records(
        gconprod_records=gconprod,
        gconinje_records=gconinje,
        unit_system=deck_file.unit_system,
        current_time=current_time,
    )
    gecon = deck_file.get("GECON") or []
    if gecon:
        apply_group_economic_limits(
            group_controls=controls,
            gecon_records=gecon,
            unit_system=deck_file.unit_system,
            current_time=current_time,
        )
    return controls


FIELD_SUMMARY_VECTORS: dict[str, type[ws.FieldRate]] = {
    "FOPR": ws.FieldOilProductionRate,
    "FWPR": ws.FieldWaterProductionRate,
    "FGPR": ws.FieldGasProductionRate,
    "FLPR": ws.FieldLiquidProductionRate,
    "FWIR": ws.FieldWaterInjectionRate,
    "FGIR": ws.FieldGasInjectionRate,
    "FOIR": ws.FieldOilInjectionRate,
    "FWCT": ws.FieldWaterCut,
    "FGOR": ws.FieldGasOilRatio,
    "FOPT": ws.FieldOilProductionTotal,
    "FWPT": ws.FieldWaterProductionTotal,
    "FGPT": ws.FieldGasProductionTotal,
    "FWIT": ws.FieldWaterInjectionTotal,
    "FGIT": ws.FieldGasInjectionTotal,
    "FOIT": ws.FieldOilInjectionTotal,
}
"""Deck mnemonic to the `bores.wells.summary` class it requests. Field-wide, no object list."""

WELL_SUMMARY_VECTORS: dict[str, type[ws.WellVector]] = {
    "WOPR": ws.WellOilProductionRate,
    "WWPR": ws.WellWaterProductionRate,
    "WGPR": ws.WellGasProductionRate,
    "WLPR": ws.WellLiquidProductionRate,
    "WWIR": ws.WellWaterInjectionRate,
    "WGIR": ws.WellGasInjectionRate,
    "WOIR": ws.WellOilInjectionRate,
    "WWCT": ws.WellWaterCut,
    "WGOR": ws.WellGasOilRatio,
    "WBHP": ws.WellBottomHolePressure,
    "WTHP": ws.WellTubingHeadPressure,
    "WOPT": ws.WellOilProductionTotal,
    "WWPT": ws.WellWaterProductionTotal,
    "WGPT": ws.WellGasProductionTotal,
    "WWIT": ws.WellWaterInjectionTotal,
    "WGIT": ws.WellGasInjectionTotal,
    "WOIT": ws.WellOilInjectionTotal,
}
"""Deck mnemonic to the `bores.wells.summary` class it requests. One instance per named well."""

REGION_SUMMARY_VECTORS: dict[str, type[ws.RegionInPlace]] = {
    "ROIP": ws.RegionOilInPlace,
    "RGIP": ws.RegionGasInPlace,
    "RWIP": ws.RegionWaterInPlace,
}
"""Deck mnemonic to the `bores.wells.summary` class it requests. One instance per `FIPNUM` region."""

BLOCK_SUMMARY_VECTORS: dict[str, type[ws.BlockVector]] = {
    "BPR": ws.BlockPressure,
    "BPRESSUR": ws.BlockPressure,
    "BGSAT": ws.BlockGasSaturation,
    "BSGAS": ws.BlockGasSaturation,
    "BOSAT": ws.BlockOilSaturation,
    "BSOIL": ws.BlockOilSaturation,
    "BWSAT": ws.BlockWaterSaturation,
    "BSWAT": ws.BlockWaterSaturation,
    "BRS": ws.BlockGasOilRatio,
    "BRV": ws.BlockVaporizedOilGasRatio,
    "BOKR": ws.BlockOilRelativePermeability,
    "BWKR": ws.BlockWaterRelativePermeability,
    "BGKR": ws.BlockGasRelativePermeability,
}
"""
Deck mnemonic to the `bores.wells.summary` class it requests. One instance per listed cell.
Mnemonics that Eclipse treats as the same vector (`BPR` and `BPRESSUR`) map to the same class.
"""


def get_fluid_in_place_regions(deck_file: DeckFile) -> list[int]:
    """
    Every fluid-in-place region number a deck defines.

    :param deck_file: The deck to read `FIPNUM` from.
    :returns: The distinct `FIPNUM` values in ascending order, or `[1]` if the deck has none.
    """
    fip_numbers = deck_file.get("FIPNUM")
    if fip_numbers is None:
        return [1]
    return sorted(int(region) for region in np.unique(np.asarray(fip_numbers)))


def load_summary_items(
    deck_file: DeckFile, *, compiled_at: float = 0.0, unit_system: UnitSystem | None = None
) -> list[ScheduleItem["CompiledBlackOilModel"]]:
    """
    Builds one `ScheduleItem` per report time from every summary-vector
    keyword in a deck's `SUMMARY` section, for report times strictly
    after `compiled_at`.

    Every requested vector is recorded together at every report time (a
    `TSTEP`/`DATES` boundary), matching Eclipse's own default summary
    output behavior. A `SUMMARY` declaration has no cadence of its own,
    unlike a `SCHEDULE`-section keyword's `schedule_time`. A well
    keyword with no object list (`WOPR` alone, say) requests every well
    ever mentioned via `WELSPECS`, matching Eclipse's own default.

    `ROIP`/`RGIP`/`RWIP` (region in-place volumes) request one vector per listed fluid-in-place
    region. With no region list they request every `FIPNUM` region in the deck (region 1 if the
    deck has no `FIPNUM`), matching Eclipse's own default.

    `BPR`, `BGSAT`, `BOSAT`, `BWSAT`, `BRS`, `BRV`, `BOKR`, `BWKR`, `BGKR` (and `BPRESSUR`, `BSGAS`,
    `BSOIL`, `BSWAT`, the names Eclipse gives the same vectors) request one vector per listed cell.
    A cell listed more than once for the same vector is recorded once.

    :param deck_file: The deck to read the `SUMMARY` section from.
    :param compiled_at: The point on the schedule clock the model this
        schedule will run against was compiled at, in the deck's time
        unit. Only report times strictly after this become items.
    :param unit_system: The unit system the items' times are expressed in. Defaults to the
        deck's own.
    :returns: One `ScheduleItem` per qualifying report time, each
        wrapping the same `RecordSummary` action, carrying every
        requested vector.
    :raises DeckParseError: If a block vector lists a cell outside the deck's grid.
    """
    well_names = [record["well"] for record in deck_file.get("WELSPECS") or []]

    quantities: list[Summary[CompiledBlackOilModel]] = []
    for mnemonic, field_summary_cls in FIELD_SUMMARY_VECTORS.items():
        if deck_file.get(mnemonic) is not None:
            quantities.append(field_summary_cls())

    for mnemonic, well_summary_cls in WELL_SUMMARY_VECTORS.items():
        requested = deck_file.get(mnemonic)
        if requested is None:
            continue
        quantities.extend(well_summary_cls(well_name=name) for name in (requested or well_names))

    for mnemonic, region_summary_cls in REGION_SUMMARY_VECTORS.items():
        requested = deck_file.get(mnemonic)
        if requested is None:
            continue
        regions = [int(region) for region in requested] or get_fluid_in_place_regions(deck_file)
        quantities.extend(region_summary_cls(region=region) for region in regions)

    dimensions = deck_file.dimensions
    block_keys: set[str] = set()
    for mnemonic, block_summary_cls in BLOCK_SUMMARY_VECTORS.items():
        requested = deck_file.get(mnemonic)
        if requested is None:
            continue
        if dimensions is None:
            raise DeckParseError(
                f"{mnemonic}: cannot place block vectors without grid dimensions "
                "(`SPECGRID`/`DIMENS` not found)."
            )
        for cell in requested:
            i, j, k = cell["i"], cell["j"], cell["k"]
            if not (
                1 <= i <= dimensions.nx and 1 <= j <= dimensions.ny and 1 <= k <= dimensions.nz
            ):
                raise DeckParseError(
                    f"{mnemonic}: cell ({i}, {j}, {k}) is outside the {dimensions.nx} x "
                    f"{dimensions.ny} x {dimensions.nz} grid."
                )
            quantity = block_summary_cls(i=i - 1, j=j - 1, k=k - 1)
            # A cell listed twice, or under two names for the same vector, is recorded once.
            if quantity.key in block_keys:
                continue
            block_keys.add(quantity.key)
            quantities.append(quantity)

    if not quantities:
        return []

    action = RecordSummary(quantities=quantities)
    time_factor = get_conversion_factors(
        deck_file.unit_system, unit_system or deck_file.unit_system
    )["time"]
    return [
        ScheduleItem(event=TimeEvent(at=time * time_factor), action=action, name=f"summary@{time}")
        for time in deck_file.report_times
        if time > compiled_at
    ]


def load_report_times(deck_file: DeckFile) -> TimeTargets:
    """
    Every report time a deck asks for, as times a time stepper must land on, so the run reports
    where the deck's `TSTEP` entries and `DATES` say it should. Combine with
    `Schedule.time_targets` for the full set a `Timer` needs.

    :param deck_file: The deck to read `TSTEP` and `DATES` from.
    :returns: The deck's report times in the deck's own time unit. The last of them is where
        the deck's schedule ends, which is the natural `simulation_time` of the run.
    """
    return TimeTargets(times=deck_file.report_times)


WLIFT_TRIGGER_QUANTITIES: dict[str, RateQuantity] = {
    "OIL": RateQuantity.OIL,
    "GAS": RateQuantity.GAS,
    "WATER": RateQuantity.WATER,
    "LIQ": RateQuantity.LIQUID,
}
"""Deck `WLIFT` trigger phase to the rate quantity its rate trigger watches."""


def get_item_start_time(item: ScheduleItem["CompiledBlackOilModel"]) -> Number:
    """
    The elapsed time a schedule item first becomes live.

    :param item: A schedule item whose event is a `TimeEvent`, an interval event, or a
        `TimeWindowEvent` combined with a condition by `AllOf`.
    :returns: The event's time, or the start of its time window.
    """
    event = item.event
    if isinstance(event, AllOf):
        event = next(inner for inner in event.events if isinstance(inner, TimeWindowEvent))
    if isinstance(event, TimeWindowEvent):
        return event.start
    return event.at  # type: ignore[attr-defined]


def load_schedule(
    deck_file: DeckFile,
    *,
    compiled_at: float = 0.0,
    load_summary: bool = True,
    unit_system: UnitSystem | None = None,
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
    `compiled_at` becomes an `ActivateWells` action, and a `COMPDAT` after
    `compiled_at` becomes an `ActivateCompletions` action. Both act on rows
    that already exist, and allocate no new ones. When the `COMPDAT` re-enters a
    connection the well already has, the action takes that connection's earlier
    version out of effect and puts the new one in. Nothing here reloads or
    recompiles anything.

    Every `WECON` record with `schedule_time <= compiled_at` is assumed
    already baked into the model's economic limits at compile time (see
    `apply_economic_limits`), and is skipped. A `WECON` reissue after
    `compiled_at` becomes one `SetLimits` action per quantity the record
    defines (a single record can set several at once, a max water cut
    and a max GOR together, say). `GECON` (a group's own economic
    limits) works the same way, one `SetGroupLimit` action per quantity,
    already-compiled rows only.

    A `WCONPROD`/`WCONINJE` reissue sets the well's control mode and
    targets. When its own `bhp`/`thp` item is given and the mode isn't
    that item, it also patches the well's implicit `BHPLimit`/`THPLimit`
    via an additional `SetLimits` action.

    Every `SetLimits`/`SetGroupLimit` action here can only patch a limit
    row the well/group already had for that kind/quantity at compile
    time. Neither `CompiledLimits`' nor `CompiledGroupLimits`' CSR table
    can grow a new row mid-schedule. `RateLimit` has no deck-record
    source in this codebase at all, so it's never emitted here.

    A `WLIFT` item with trigger conditions holds across checks, so it runs once by default.
    Advance the returned schedule with a `Firings` to count that.

    The status item of a `WCONPROD`/`WCONINJE` record, and a `WELOPEN` record that names no
    connection, set the well's own open or shut flag (`UpdateWellStatuses`). A `WELOPEN` that names
    connections sets only those connections' status. A `WCONPROD`/`WCONINJE` status of `OPEN`,
    which is also what an omitted status means, opens the well again if it was shut.

    Items that fall at the same time run in a fixed order, by the keyword they came from:
    `WELSPECS`, `WPIMULT`, `WLIFT`, `WCONPROD`, `WCONINJE`, `WELTARG`, `WECON`, `GECON`,
    `COMPDAT`, `WELOPEN`, then summary recording. Records of the same keyword at the same
    time run in deck order. A `WELTARG` therefore fine-tunes the control a `WCONPROD` or
    `WCONINJE` at the same time sets, which is also how `load_controls_from_records` replays
    them. A `WELOPEN` comes after the `WCONPROD`/`WCONINJE` status and after `COMPDAT`, so it
    has the last word on whether a well is open, and can target connections that a `COMPDAT`
    at the same time brings into effect. Summary vectors are recorded last, after every change
    at that time. The order is not the order different keywords appear in the deck.

    When `load_summary` is `True`, every requested `SUMMARY`-section vector
    (see `load_summary_items`) is added too, one item per report time
    strictly after `compiled_at`.

    :param deck_file: The deck to read schedule keywords from.
    :param compiled_at: The point on the schedule clock the model this
        schedule will run against was compiled at, in the deck's time
        unit. Only events strictly after this become actions.
    :param load_summary: Whether to add scheduled summary-vector recording
        items from the deck's `SUMMARY` section.
    :param unit_system: The unit system the schedule will run against. Every event time and every
        rate, pressure and ratio the actions carry is expressed in it. Defaults to the deck's own.
    :returns: A `Schedule[CompiledBlackOilModel]`, one rule per
        qualifying record, sorted by `schedule_time`.
    """
    deck_unit_system = deck_file.unit_system
    target_unit_system = unit_system if unit_system is not None else deck_unit_system
    factors = get_conversion_factors(deck_unit_system, target_unit_system)
    time_factor = factors["time"]
    items: list[ScheduleItem[CompiledBlackOilModel]] = []
    empty: list[dict[str, typing.Any]] = []

    def is_due(record: typing.Mapping[str, typing.Any]) -> bool:
        return record.get("schedule_time", 0.0) > compiled_at

    for record in deck_file.get("WELSPECS") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = ActivateWells(well_names=(record["well"],))
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=action,
                name=f"welspecs:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WPIMULT") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        wpimult_layers = connection_layers(record, keyword="WPIMULT")
        action = MultiplyConnectionFactors(
            well_names=(record["well"],),
            multiplier=record["multiplier"],
            i=record.get("i", 0),
            j=record.get("j", 0),
            k1=wpimult_layers[0],
            k2=wpimult_layers[1],
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=action,
                name=f"wpimult:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WLIFT") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        well_name = record["well"]
        unsupported = [
            name
            for name, value in (
                ("`alq_shift`", record["alq_shift"]),
                ("`thp_shift`", record["thp_shift"]),
            )
            if not np.isclose(value, 1.0e20)
        ]
        if unsupported:
            raise NotSupportedError(
                f"`WLIFT` for well {well_name!r} uses {', '.join(unsupported)}, which is not "
                "supported yet."
            )

        actions: list[Action] = [
            SetWellLift(
                well_name=well_name,
                vfp_table=record["new_vfp_table"] or None,
                artificial_lift_quantity=record["new_alq"] or None,
                efficiency_factor=record["new_efficiency_factor"] or None,
            )
        ]
        if not np.isclose(record["new_thp_limit"], 0.0):
            actions.append(
                SetLimits(
                    well_names=(well_name,),
                    kinds=(LimitKind.THP,),
                    min_values=record["new_thp_limit"] * factors["pressure"],
                )
            )
        action = actions[0] if len(actions) == 1 else RunSequence(actions=actions)

        triggers: list[Event] = []
        if not np.isclose(record["trigger_limit"], 0.0):
            triggers.append(
                RateThreshold(
                    well_name=well_name,
                    quantity=WLIFT_TRIGGER_QUANTITIES[record["trigger_phase"]],
                    surface=True,
                    op=ComparisonOperator.LT,
                    threshold=record["trigger_limit"]
                    * (
                        factors["gas_surface_rate"]
                        if WLIFT_TRIGGER_QUANTITIES[record["trigger_phase"]] is RateQuantity.GAS
                        else factors["liquid_surface_rate"]
                    ),
                )
            )
        if not np.isclose(record["water_cut_limit"], 0.0):
            triggers.append(
                WaterCutThreshold(
                    well_name=well_name,
                    op=ComparisonOperator.GT,
                    threshold=record["water_cut_limit"],
                )
            )
        if not np.isclose(record["gas_oil_ratio_limit"], 0.0):
            triggers.append(
                GasOilRatioThreshold(
                    well_name=well_name,
                    op=ComparisonOperator.GT,
                    threshold=record["gas_oil_ratio_limit"] * factors["gas_oil_ratio"],
                )
            )

        event: Event
        if not triggers:
            event = TimeEvent(at=schedule_time * time_factor)
        else:
            # A trigger is live from this record until the deck next changes to the well's lift
            # settings (a later `WLIFT` or `WCONPROD` for it). The new settings persist in the
            # compiled arrays once applied, and re-applying them while the trigger holds is
            # idempotent, so no one-shot state is needed.
            later_changes = [
                other["schedule_time"]
                for keyword in ("WLIFT", "WCONPROD")
                for other in deck_file.get(keyword) or empty
                if other["well"] == well_name and other["schedule_time"] > schedule_time
            ]
            event = AllOf(
                events=(
                    TimeWindowEvent(
                        start=schedule_time * time_factor,
                        end=min(later_changes) * time_factor if later_changes else None,
                    ),
                    triggers[0] if len(triggers) == 1 else AnyOf(events=triggers),
                )
            )
        items.append(
            ScheduleItem(
                event=event,
                action=action,
                name=f"wlift:{well_name}@{schedule_time}",
            )
        )

    for record in deck_file.get("WCONPROD") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        control = load_producer_control_from_record(record, unit_system=deck_unit_system).convert(
            target_unit_system
        )
        action = SetWellControls(
            well_names=(record["well"],),
            mode=control.mode,
            target_rate=control.target_rate,
            target_bhp=control.target_bhp,
            target_thp=control.target_thp,
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=action,
                name=f"wconprod:{record['well']}@{schedule_time}",
            )
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=SetWellLift(
                    well_name=record["well"],
                    vfp_table=control.vfp_table or 0,
                    artificial_lift_quantity=control.artificial_lift_quantity,
                ),
                name=f"wconprod-lift:{record['well']}@{schedule_time}",
            )
        )
        for limit in control.limits:
            if isinstance(limit, BHPLimit):
                kind = LimitKind.BHP
            elif isinstance(limit, THPLimit):
                kind = LimitKind.THP
            else:
                continue
            limit_action = SetLimits(
                well_names=(record["well"],),
                kinds=(kind,),
                min_values=limit.min_value,
                max_values=limit.max_value,
            )
            items.append(
                ScheduleItem(
                    event=TimeEvent(at=schedule_time * time_factor),
                    action=limit_action,
                    name=f"wconprod-{kind.name.lower()}limit:{record['well']}@{schedule_time}",
                )
            )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=UpdateWellStatuses(
                    well_names=(record["well"],),
                    status=WELOPEN_STATUS_MAP[str(record.get("status") or "OPEN").upper()],
                ),
                name=f"wconprod-status:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WCONINJE") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        control = load_injector_control_from_record(record, unit_system=deck_unit_system).convert(
            target_unit_system
        )
        action = SetWellControls(
            well_names=(record["well"],),
            mode=control.mode,
            target_rate=control.target_rate,
            target_bhp=control.target_bhp,
            target_thp=control.target_thp,
            injected_phase=control.injected_phase,
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=action,
                name=f"wconinje:{record['well']}@{schedule_time}",
            )
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=SetWellLift(well_name=record["well"], vfp_table=control.vfp_table or 0),
                name=f"wconinje-lift:{record['well']}@{schedule_time}",
            )
        )
        for limit in control.limits:
            if isinstance(limit, BHPLimit):
                kind = LimitKind.BHP
            elif isinstance(limit, THPLimit):
                kind = LimitKind.THP
            else:
                continue
            limit_action = SetLimits(
                well_names=(record["well"],),
                kinds=(kind,),
                min_values=limit.min_value,
                max_values=limit.max_value,
            )
            items.append(
                ScheduleItem(
                    event=TimeEvent(at=schedule_time * time_factor),
                    action=limit_action,
                    name=f"wconinje-{kind.name.lower()}limit:{record['well']}@{schedule_time}",
                )
            )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=UpdateWellStatuses(
                    well_names=(record["well"],),
                    status=WELOPEN_STATUS_MAP[str(record.get("status") or "OPEN").upper()],
                ),
                name=f"wconinje-status:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WELTARG") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = SetWellTargets(
            well_names=(record["well"],),
            control_mode=WellTargetMode(record["control_mode"]),
            value=record.get("value"),
            unit_system=deck_unit_system,
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=action,
                name=f"weltarg:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WECON") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        for limit in (
            limit.convert(target_unit_system)
            for limit in load_economic_limits_from_record(record, unit_system=deck_unit_system)
        ):
            action = SetLimits(
                well_names=(record["well"],),
                kinds=(LimitKind.ECONOMIC,),
                quantities=limit.quantity,
                min_values=limit.min_value,
                max_values=limit.max_value,
                workover_actions=limit.workover_action,
                end_runs=limit.end_run,
            )
            items.append(
                ScheduleItem(
                    event=TimeEvent(at=schedule_time * time_factor),
                    action=action,
                    name=f"wecon:{record['well']}:{limit.quantity}@{schedule_time}",
                )
            )

    for record in deck_file.get("GECON") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        for limit in (
            limit.convert(target_unit_system)
            for limit in load_group_economic_limits_from_record(
                record, unit_system=deck_unit_system
            )
        ):
            action = SetGroupLimit(
                group_name=record["group"],
                quantity=limit.quantity,
                min_value=limit.min_value,
                max_value=limit.max_value,
                workover_action=limit.workover_action,
                end_run=limit.end_run,
            )
            items.append(
                ScheduleItem(
                    event=TimeEvent(at=schedule_time * time_factor),
                    action=action,
                    name=f"gecon:{record['group']}:{limit.quantity}@{schedule_time}",
                )
            )

    for record in deck_file.get("COMPDAT") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        action = ActivateCompletions(
            well_names=(record["well"],),
            i=record["i"],
            j=record["j"],
            k1=record["k1"],
            k2=record["k2"],
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=action,
                name=f"compdat:{record['well']}@{schedule_time}",
            )
        )

    for record in deck_file.get("WELOPEN") or empty:
        if not is_due(record):
            continue
        schedule_time = record["schedule_time"]
        welopen_layers = connection_layers(record, keyword="WELOPEN")
        action = UpdateWellStatuses(
            well_names=(record["well"],),
            status=WELOPEN_STATUS_MAP[record["status"]],
            i=record.get("i", 0),
            j=record.get("j", 0),
            k1=welopen_layers[0],
            k2=welopen_layers[1],
        )
        items.append(
            ScheduleItem(
                event=TimeEvent(at=schedule_time * time_factor),
                action=action,
                name=f"welopen:{record['well']}@{schedule_time}",
            )
        )

    if load_summary:
        items.extend(
            load_summary_items(deck_file, compiled_at=compiled_at, unit_system=target_unit_system)
        )
    items.sort(key=get_item_start_time)
    return Schedule(items=tuple(items))
