"""Well-specific `Event`/`Action` implementations, built on `bores.schedule`."""

import enum
import typing

import attrs
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.schedule.actions import RunSequence
from bores.schedule.base import Action, ScheduleContext
from bores.schedule.events import ThresholdEvent
from bores.types import (
    Boolean,
    FluidPhase,
    IntArray,
    Integer,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
)
from bores.wells.base import CompletionStatus, WellStatus
from bores.wells.compile import (
    UNSET_INT,
    CompiledGroupLimits,
    CompiledLimits,
    CompiledPerforations,
    CompiledWellControls,
    CompiledWellSystem,
    LimitKind,
    WellKind,
    get_well_status_tag,
)
from bores.wells.controls import (
    EconomicQuantity,
    InjectorControlMode,
    ProducerControlMode,
    RateQuantity,
    WellTargetMode,
    WorkoverAction,
)
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.mappings import (
    INJECTOR_CONTROL_MODE_MAP,
    PRODUCER_CONTROL_MODE_MAP,
    WELTARG_TARGET_FIELD,
)
from bores.wells.workspace import WellsWorkspace

if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel
    from bores.schedule.base import Schedule
    from bores.wells.model import WellSystem


__all__ = [
    "RATE_ARRAYS",
    "TARGET_SETTERS",
    "ActivateCompletions",
    "ActivateWells",
    "ApplicationMode",
    "GasOilRatioThreshold",
    "MultiplyConnectionFactors",
    "OpenWells",
    "RateThreshold",
    "SetGroupLimit",
    "SetLimits",
    "SetWellControls",
    "SetWellLift",
    "SetWellTargets",
    "ShutInWells",
    "UpdateWellStatuses",
    "WaterCutThreshold",
    "as_number_or_array",
    "broadcast_or_match",
    "check_vfp_references",
    "expand_values",
    "get_many_matching_connection_rows",
    "get_matching_connection_rows",
    "match_connection_target",
    "resolve_group",
    "resolve_well",
    "resolve_wells",
]


def resolve_well(
    *, model: "CompiledBlackOilModel", well_name: str
) -> tuple[Integer, CompiledWellSystem]:
    """
    Finds and returns a well's row and its compiled well system within a model.

    :param model: The model to look `well_name` up in.
    :param well_name: The well to find.
    :returns: `(well_row, model.wells)`.
    :raises ValidationError: If `model.wells` is `None`.
    """
    if model.wells is None:
        raise ValidationError(f"{model!r} has no compiled wells.")
    return typing.cast(Integer, model.wells.well_row(name=well_name)), model.wells


def resolve_group(
    *, model: "CompiledBlackOilModel", group_name: str
) -> tuple[Integer, CompiledWellSystem]:
    """
    Finds a group's row and its compiled well system within a model.

    :param model: The model to look `group_name` up in.
    :param group_name: The group to find.
    :returns: `(group_row, model.wells)`.
    :raises ValidationError: If `model.wells` is `None`, or it has no `group_controls`.
    """
    if model.wells is None:
        raise ValidationError(f"{model!r} has no compiled wells.")
    if model.wells.group_controls is None:
        raise ValidationError(f"{model!r}'s compiled wells have no group controls.")
    try:
        return model.wells.group_controls.names.index(group_name), model.wells
    except ValueError:
        raise ValidationError(f"No group named {group_name!r} in this compiled system.") from None


def resolve_wells(
    *, model: "CompiledBlackOilModel", well_names: typing.Sequence[str]
) -> tuple[IntArray[OneDimension], CompiledWellSystem]:
    """
    Finds and returns several wells' rows and the shared compiled well system within a model.

    :param model: The model to look `well_names` up in.
    :param well_names: The wells to find.
    :returns: `(well_rows, model.wells)`, `well_rows` in the same order as `well_names`.
    :raises ValidationError: If `model.wells` is `None`.
    """
    if model.wells is None:
        raise ValidationError(f"{model!r} has no compiled wells.")
    well_rows = np.asarray(model.wells.well_row(name=well_names), dtype=np.intp)
    return well_rows, model.wells  # type: ignore[return-value]


BulkValue = typing.TypeVar("BulkValue")
"""One bulk action field's own value type, e.g. `CompletionStatus` or `Number`."""


def broadcast_or_match(
    value: BulkValue | tuple[BulkValue, ...], *, count: Integer, field_name: str
) -> BulkValue | tuple[BulkValue, ...]:
    """
    Accepts a bulk action field: a single value to broadcast to every
    target, or a sequence already aligned one-to-one with the targets.

    :param value: The value as given to the action.
    :param count: How many targets this field must align with, if given as a sequence.
    :param field_name: The field's own name, for a validation message.
    :returns: `value` unchanged; a sequence is only length-checked, not converted.
    :raises ValidationError: If `value` is a sequence whose length isn't `count`.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        return value
    if len(value) != count:
        raise ValidationError(
            f"{field_name!r} was given {len(value)} value(s) for {count} target(s). Give one "
            "value to broadcast to every target, or exactly one per target."
        )
    return value


def as_number_or_array(value: Number | typing.Sequence[Number]) -> NumberOrArray[OneDimension]:
    """
    Normalizes a scalar or per-target numeric value into the array-like
    form expected by bulk numeric setters.

    :param value: A single numeric value, or a sequence of numeric values
        that has already been validated against the target count.
    :returns: The original scalar if one was provided; otherwise, the
        values converted to a NumPy array.
    """
    if isinstance(value, (list, tuple)):
        return np.asarray(value)  # type: ignore[return-value]
    return typing.cast(NumberOrArray[OneDimension], value)


def expand_values(
    values: typing.Sequence[BulkValue] | NumberArray[OneDimension], *, counts: typing.Sequence[int]
) -> list[BulkValue]:
    """
    Repeats each input value according to a matching count so the
    resulting list lines up with a concatenated target array.

    :param values: One value per item in the original grouping, such as
        a `tuple`/`list` returned by a broadcast-or-match helper or a
        direct `np.ndarray` supplied by a caller.
    :param counts: How many times each corresponding value should be
        repeated, in the same order as `values`.
    :returns: A flat list where each `values[i]` is repeated
        `counts[i]` times and concatenated in order.
    """
    expanded: list[BulkValue] = []
    for value, count in zip(values, counts, strict=True):
        expanded.extend([typing.cast(BulkValue, value)] * count)
    return expanded


def match_connection_target(
    *,
    i: Integer,
    j: Integer,
    k1: Integer,
    k2: Integer,
    connection_i: Integer | IntArray[OneDimension],
    connection_j: Integer | IntArray[OneDimension],
    connection_k1: Integer | IntArray[OneDimension],
    connection_k2: Integer | IntArray[OneDimension],
) -> typing.Any:
    """
    Tells which connections a deck-style `(i, j, k1, k2)` target selects.

    A `0` for `i` or `j` matches any column. `k1` and `k2` give an inclusive layer range that
    selects every connection whose own layers overlap it. Both `0` means any layer. All indices
    are 1-based, as in a deck.

    :param i: The target's I index, or `0` for any.
    :param j: The target's J index, or `0` for any.
    :param k1: The first layer of the target's range, or `0` for any layer.
    :param k2: The last layer of the target's range, or `0` for any layer.
    :param connection_i: Each connection's I index.
    :param connection_j: Each connection's J index.
    :param connection_k1: Each connection's first layer.
    :param connection_k2: Each connection's last layer.
    :returns: Whether each connection is selected, a single boolean for a single connection.
    :raises ValidationError: If the layer range is given only in part, or is reversed.
    """
    if (k1 == 0) != (k2 == 0) or (k1 and k1 > k2):
        raise ValidationError(
            f"The layer range k1={k1}, k2={k2} is invalid. Give both as 1-based layers with "
            "`k1 <= k2`, or both as `0` for every layer."
        )

    selected = np.ones(np.shape(connection_i), dtype=bool)
    if i:
        selected &= np.asarray(connection_i) == i
    if j:
        selected &= np.asarray(connection_j) == j
    if k1:
        selected &= (np.asarray(connection_k1) <= k2) & (np.asarray(connection_k2) >= k1)
    return selected[()]


def get_matching_connection_rows(
    *,
    perforations: CompiledPerforations,
    well_row: Integer,
    i: Integer,
    j: Integer,
    k1: Integer,
    k2: Integer,
    grid: Grid | None = None,
    well_name: str | None = None,
    require_match: Boolean = True,
    active_only: Boolean = False,
) -> IntArray[OneDimension]:
    """
    Selects the connections of a well that a deck-style `(i, j, k1, k2)` target names.

    `(0, 0, 0, 0)` selects every connection. Otherwise a `0` for `i` or `j` matches any
    column, and `k1`/`k2` give an inclusive layer range, which must either both be given
    or both be `0` (any layer). Indices are 1-based, as in a deck.

    :param perforations: `CompiledPerforations` for the whole system.
    :param well_row: The well's row.
    :param i: 1-based I index, or `0` for any.
    :param j: 1-based J index, or `0` for any.
    :param k1: 1-based first layer of the range, or `0` for any layer.
    :param k2: 1-based last layer of the range, or `0` for any layer.
    :param grid: Required when `(i, j, k1, k2)` targets specific connections.
    :param well_name: The well's name, for an error message.
    :param require_match: Whether a target that selects no connection is an error.
    :param active_only: Whether to leave out connections not yet in effect, such as one that
        starts later or one a later `COMPDAT` has replaced.
    :returns: The selected connection rows.
    :raises ValidationError: If connections are targeted but `grid` wasn't given, the layer
        range is incomplete or reversed, or `require_match` is set and nothing is selected.
    """
    all_rows = perforations.connection_rows(well_row=well_row)
    rows = np.arange(all_rows.start, all_rows.stop)
    if active_only:
        rows = rows[perforations.schedule_statuses[rows] == get_well_status_tag(WellStatus.ACTIVE)]
    if i == 0 and j == 0 and k1 == 0 and k2 == 0:
        return typing.cast(IntArray[OneDimension], rows)

    if grid is None or grid.dimensions is None:
        raise ValidationError("A grid with dimensions is required to target specific connections.")

    dims = grid.dimensions
    cells = perforations.get_cell_index(row=typing.cast(IntArray[OneDimension], rows))
    layers = cells // (dims.nx * dims.ny) + 1
    mask = match_connection_target(
        i=i,
        j=j,
        k1=k1,
        k2=k2,
        connection_i=cells % dims.nx + 1,  # type: ignore[arg-type]
        connection_j=(cells // dims.nx) % dims.ny + 1,  # type: ignore[arg-type]
        connection_k1=layers,  # type: ignore[arg-type]
        connection_k2=layers,  # type: ignore[arg-type]
    )
    selected = typing.cast(IntArray[OneDimension], rows[mask])
    if require_match and selected.size == 0:
        raise ValidationError(
            f"Well {well_name if well_name is not None else well_row!r} has no connection "
            f"at i={i}, j={j}, k1={k1}, k2={k2}."
        )
    return selected


def get_many_matching_connection_rows(
    *,
    perforations: CompiledPerforations,
    well_rows: IntArray[OneDimension],
    i: Integer,
    j: Integer,
    k1: Integer,
    k2: Integer,
    grid: Grid | None,
    well_names: typing.Sequence[str] | None = None,
    active_only: Boolean = False,
) -> tuple[IntArray[OneDimension], list[int]]:
    """
    Selects matching connections across several wells at once.

    A well with no connection at the target simply contributes none, but it is an
    error if no well at all has one.

    :param perforations: `CompiledPerforations` for the whole system.
    :param well_rows: Every well's own row.
    :param i: 1-based I index, or `0` for any.
    :param j: 1-based J index, or `0` for any.
    :param k1: 1-based first layer of the range, or `0` for any layer.
    :param k2: 1-based last layer of the range, or `0` for any layer.
    :param grid: Required when `(i, j, k1, k2)` targets specific connections.
    :param well_names: The wells' names, for an error message.
    :param active_only: Whether to leave out connections not yet in effect.
    :returns: Every well's selected rows, concatenated in `well_rows`' order, and, in the
        same order, how many rows each well contributed (to later align a per-well value
        with the concatenated rows via `expand_values`).
    :raises ValidationError: If the target is invalid, or selects no connection on any well.
    """
    rows_per_well = [
        get_matching_connection_rows(
            perforations=perforations,
            well_row=well_row,
            i=i,
            j=j,
            k1=k1,
            k2=k2,
            grid=grid,
            require_match=False,
            active_only=active_only,
        )
        for well_row in well_rows
    ]
    counts = [len(rows) for rows in rows_per_well]
    if not rows_per_well:
        return np.empty(0, dtype=np.intp), counts  # type: ignore[return-value]
    if sum(counts) == 0:
        raise ValidationError(
            f"None of the wells {list(well_names) if well_names is not None else ''} has a "
            f"connection at i={i}, j={j}, k1={k1}, k2={k2}."
        )
    return np.concatenate(rows_per_well), counts  # type: ignore[return-value]


TARGET_SETTERS = {
    "target_rate": lambda controls, well_row, value: controls.set_target_rate(
        well_row=well_row, value=value
    ),
    "target_bhp": lambda controls, well_row, value: controls.set_target_bhp(
        well_row=well_row, value=value
    ),
    "target_thp": lambda controls, well_row, value: controls.set_target_thp(
        well_row=well_row, value=value
    ),
}
"""Maps a `WELTARG_TARGET_FIELD` entry to the `CompiledWellControls` setter it calls."""


def check_control_mode_kind(
    *,
    controls: CompiledWellControls,
    well_rows: typing.Sequence[Integer],
    well_names: typing.Sequence[str],
    modes: typing.Sequence[ProducerControlMode | InjectorControlMode],
) -> None:
    """
    Checks that every well's control mode is one its own kind (producer or injector) can use.

    :param controls: The compiled well controls.
    :param well_rows: Each well's row.
    :param well_names: Each well's name, for the error message.
    :param modes: Each well's new control mode.
    :raises ValidationError: If a mode does not suit its well's kind. Nothing has been written.
    """
    for well_row, well_name, mode in zip(well_rows, well_names, modes, strict=True):
        is_injector = controls.well_kinds[well_row] == WellKind.INJECTOR
        expected = InjectorControlMode if is_injector else ProducerControlMode
        if not isinstance(mode, expected):
            raise ValidationError(
                f"Control mode {mode!r} doesn't apply to "
                f"{'an injector' if is_injector else 'a producer'} ({well_name!r})."
            )


class ApplicationMode(enum.Enum):
    """How `SetLimits`'s well-targeting field pairs with its limit-targeting field(s)."""

    ONE_TO_ONE = "one_to_one"
    """
    Each index across `well_names` and `kinds`/`quantities` names one
    distinct target; all three must be the same length.
    """

    ALL = "all"
    """
    Every entry in `kinds`/`quantities` applies to every well in
    `well_names`, a full cross product.
    """


@attrs.frozen(kw_only=True, slots=True)
class ActivateWells(Action["CompiledBlackOilModel"]):
    """
    Brings wells online, changing their schedule status from `PENDING` to `ACTIVE`. A deck
    `WELSPECS` record that starts after the run begins loads as one of these.

    Only the wells themselves are activated. Each well's connections come online
    separately, through `ActivateCompletions`.
    """

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to activate."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Sets every named well's own schedule status to `ACTIVE`, in one bulk write.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with every named well activated.
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        wells.set_schedule_status(well_row=well_rows, status=WellStatus.ACTIVE)
        return model


@attrs.frozen(kw_only=True, slots=True)
class UpdateWellStatuses(Action["CompiledBlackOilModel"]):
    """
    Opens or shuts wells, or specific connections on them, all against the same targeted
    connection(s). A deck `WELOPEN` record loads as one of these.

    A change to specific connections only reaches connections already in effect. A change to
    a whole well also reaches connections that start later, so a well that is shut in stays
    shut when they come online.
    """

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to act on."""

    status: CompletionStatus | tuple[CompletionStatus, ...]
    """
    The new open/shut status. A single value applies to every well;
    a sequence matching `well_names` sets each well to its own status.
    """

    i: Integer = 0
    """The connection's grid I index (1-based), or `0` to target every connection on the well."""

    j: Integer = 0
    """The connection's grid J index (1-based), or `0`. Only meaningful together with `i`."""

    k1: Integer = 0
    """The starting grid K index (1-based) of the targeted layer range, or `0`."""

    k2: Integer = 0
    """The ending grid K index (1-based) of the targeted layer range, or `0`."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Sets the targeted connection(s)' open/shut status across every
        named well, in one bulk write.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the targeted connections patched.
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        status = broadcast_or_match(self.status, count=len(self.well_names), field_name="status")
        grid = model.reservoir.grid if (self.i or self.j or self.k1 or self.k2) else None
        rows, counts = get_many_matching_connection_rows(
            perforations=wells.perforations,
            well_rows=well_rows,
            i=self.i,
            j=self.j,
            k1=self.k1,
            k2=self.k2,
            grid=grid,
            well_names=self.well_names,
            active_only=bool(grid),
        )
        if isinstance(status, (list, tuple, np.ndarray)):
            status = expand_values(status, counts=counts)
        wells.perforations.set_completion_status(row=rows, status=status)
        return model


@attrs.frozen(kw_only=True, slots=True)
class OpenWells(UpdateWellStatuses):
    """Sets the targeted completions on several wells to `OPEN`."""

    status: CompletionStatus = attrs.field(
        default=CompletionStatus.OPEN,
        validator=attrs.validators.in_((CompletionStatus.OPEN,)),
    )


@attrs.frozen(kw_only=True, slots=True)
class ShutInWells(UpdateWellStatuses):
    """Sets the targeted completions on several wells to `SHUT`."""

    status: CompletionStatus = attrs.field(
        default=CompletionStatus.SHUT,
        validator=attrs.validators.in_((CompletionStatus.SHUT,)),
    )


@attrs.frozen(kw_only=True, slots=True)
class MultiplyConnectionFactors(Action["CompiledBlackOilModel"]):
    """
    Multiplies the connection factors of wells, or of specific connections on them, all
    against the same targeted connection(s). A deck `WPIMULT` record loads as one of these.

    Only connections already in effect are scaled. One that starts later, or that a `COMPDAT`
    re-entry replaces, begins from its own factor.
    """

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to act on."""

    multiplier: float | tuple[float, ...]
    """
    The multiplier to apply. A single value applies to every well;
    a sequence matching `well_names` gives each well its own multiplier.
    """

    i: Integer = 0
    """The connection's grid I index (1-based), or `0` to target every connection on the well."""

    j: Integer = 0
    """The connection's grid J index (1-based), or `0`. Only meaningful together with `i`."""

    k1: Integer = 0
    """The starting grid K index (1-based) of the targeted layer range, or `0`."""

    k2: Integer = 0
    """The ending grid K index (1-based) of the targeted layer range, or `0`."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Multiplies the targeted connection(s)' connection factor across
        every named well, in one bulk write.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the targeted connections patched.
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        multiplier = broadcast_or_match(
            self.multiplier, count=len(self.well_names), field_name="multiplier"
        )
        grid = model.reservoir.grid if (self.i or self.j or self.k1 or self.k2) else None
        rows, counts = get_many_matching_connection_rows(
            perforations=wells.perforations,
            well_rows=well_rows,
            i=self.i,
            j=self.j,
            k1=self.k1,
            k2=self.k2,
            grid=grid,
            well_names=self.well_names,
            active_only=True,
        )
        if isinstance(multiplier, (list, tuple, np.ndarray)):
            multiplier = expand_values(multiplier, counts=counts)
        wells.perforations.multiply_well_index(row=rows, factor=as_number_or_array(multiplier))
        return model


@attrs.frozen(kw_only=True, slots=True)
class ActivateCompletions(Action["CompiledBlackOilModel"]):
    """
    Brings connections online, all against the same targeted connection(s). A deck `COMPDAT`
    record that starts after the run begins loads as one of these.

    A connection a well already has can be re-entered by a later `COMPDAT`. Each version of it
    then waits as `PENDING` until its own start, when this takes the version that follows the
    one in effect and puts it in effect, taking the earlier version out of effect. A re-entered
    connection therefore starts from its own definition, with no multiplier carried over.
    """

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to act on."""

    i: Integer = 0
    """The connection's grid I index (1-based), or `0` to target every connection on the well."""

    j: Integer = 0
    """The connection's grid J index (1-based), or `0`. Only meaningful together with `i`."""

    k1: Integer = 0
    """The starting grid K index (1-based) of the targeted layer range, or `0`."""

    k2: Integer = 0
    """The ending grid K index (1-based) of the targeted layer range, or `0`."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Sets the targeted connection(s)' schedule status to `ACTIVE`
        across every named well, in one bulk write.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the targeted connections activated.
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        perforations = wells.perforations
        active = get_well_status_tag(WellStatus.ACTIVE)
        grid = model.reservoir.grid if (self.i or self.j or self.k1 or self.k2) else None
        matched_any = False
        for well_row in well_rows:
            rows = get_matching_connection_rows(
                perforations=perforations,
                well_row=well_row,
                i=self.i,
                j=self.j,
                k1=self.k1,
                k2=self.k2,
                grid=grid,
                require_match=False,
            )
            if rows.size == 0:
                continue
            matched_any = True

            # Versions of one connection cover the same cells, so the cells identify it.
            ordinals = perforations.perforation_indices[rows]
            versions: dict[frozenset[Integer], dict[Integer, IntArray[OneDimension]]] = {}
            for ordinal in np.unique(ordinals):
                ordinal_rows = typing.cast(IntArray[OneDimension], rows[ordinals == ordinal])
                cells = frozenset(cell for cell in perforations.get_cell_index(row=ordinal_rows))
                versions.setdefault(cells, {})[ordinal] = ordinal_rows

            for by_ordinal in versions.values():
                in_effect = [
                    ordinal
                    for ordinal, ordinal_rows in by_ordinal.items()
                    if perforations.schedule_statuses[ordinal_rows[0]] == active
                ]
                latest_in_effect = max(in_effect, default=-1)
                waiting = [
                    ordinal
                    for ordinal, ordinal_rows in by_ordinal.items()
                    if perforations.schedule_statuses[ordinal_rows[0]] != active
                    and ordinal > latest_in_effect
                ]
                if not waiting:
                    continue
                for ordinal in in_effect:
                    perforations.set_schedule_status(
                        row=by_ordinal[ordinal], status=WellStatus.PENDING
                    )
                perforations.set_schedule_status(
                    row=by_ordinal[min(waiting)], status=WellStatus.ACTIVE
                )

        if not matched_any:
            raise ValidationError(
                f"None of the wells {list(self.well_names)} has a connection at "
                f"i={self.i}, j={self.j}, k1={self.k1}, k2={self.k2}."
            )
        return model


@attrs.frozen(kw_only=True, slots=True)
class SetWellControls(Action["CompiledBlackOilModel"]):
    """
    Changes wells' control mode and whichever targets are given. A deck `WCONPROD` or
    `WCONINJE` record loads as one of these.
    """

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to act on."""

    mode: (
        ProducerControlMode
        | InjectorControlMode
        | tuple[ProducerControlMode | InjectorControlMode, ...]
    )
    """
    The new control mode. Must match each well's own kind. A single
    value applies to every well; a sequence matching `well_names` gives
    each well its own mode.
    """

    target_rate: Number | tuple[Number, ...] | None = None
    """The new target rate(s), if given."""

    target_bhp: Number | tuple[Number, ...] | None = None
    """The new target BHP(s), if given."""

    target_thp: Number | tuple[Number, ...] | None = None
    """The new target THP(s), if given."""

    injected_phase: FluidPhase | tuple[FluidPhase, ...] | None = None
    """The new injected phase(s), if given. Only meaningful on an injector."""

    efficiency_factor: Number | tuple[Number, ...] | None = None
    """The new efficiency factor(s), if given."""

    guide_rate: Number | tuple[Number, ...] | None = None
    """The new guide rate(s), if given."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Overwrites every field given, in place, on every named well's control.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with every named well's control patched.
        :raises ValidationError: If a mode doesn't apply to its well's kind. Nothing is changed.
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        controls = wells.controls
        count = len(self.well_names)
        modes = as_list(self.mode, count=count, field_name="mode")
        check_control_mode_kind(
            controls=controls,
            well_rows=typing.cast(typing.Sequence[Integer], well_rows),
            well_names=self.well_names,
            modes=modes,
        )
        controls.set_control_mode(well_row=well_rows, mode=modes)
        if self.target_rate is not None:
            controls.set_target_rate(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.target_rate, count=count, field_name="target_rate")
                ),
            )
        if self.target_bhp is not None:
            controls.set_target_bhp(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.target_bhp, count=count, field_name="target_bhp")
                ),
            )
        if self.target_thp is not None:
            controls.set_target_thp(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.target_thp, count=count, field_name="target_thp")
                ),
            )
        if self.injected_phase is not None:
            controls.set_injected_phase(
                well_row=well_rows,
                phase=broadcast_or_match(
                    self.injected_phase, count=count, field_name="injected_phase"
                ),
            )
        if self.efficiency_factor is not None:
            controls.set_efficiency_factor(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(
                        self.efficiency_factor, count=count, field_name="efficiency_factor"
                    )
                ),
            )
        if self.guide_rate is not None:
            controls.set_guide_rate(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.guide_rate, count=count, field_name="guide_rate")
                ),
            )
        return model


@attrs.frozen(kw_only=True, slots=True)
class SetWellTargets(Action["CompiledBlackOilModel"]):
    """
    Changes wells' control mode and the one target value that mode names. A deck `WELTARG`
    record loads as one of these.

    Each well's own kind (producer or injector) picks which of `ProducerControlMode`/`InjectorControlMode`
    `control_mode` resolves to, so a mixed set of producers and injectors is fine even when
    `control_mode` is broadcast to all of them.
    """

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to act on."""

    control_mode: WellTargetMode | tuple[WellTargetMode, ...]
    """The mode(s) to switch to, as a `WellTargetMode`. A single value
    applies to every well; a sequence matching `well_names` gives each
    well its own mode."""

    value: Number | tuple[Number | None, ...] | None = None
    """The new value(s) for whichever field each well's own
    `control_mode` names. A single value applies to every well; a
    sequence matching `well_names` gives each well its own value.
    `None` (for a well, or for all of them) is only valid when that
    well's own `control_mode` is `GROUP`."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Sets every named well's control mode, and its named target value if it has one.

        Every well's own mode is resolved individually, since whether
        it translates to a `ProducerControlMode` or an
        `InjectorControlMode` depends on that well's own kind, but the
        actual value writes are still batched: every well needing the
        same target field (rate, BHP, or THP) is written in one bulk
        call, not one call per well.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with every named well's control patched.
        :raises ValidationError: If a well's `control_mode` doesn't
            apply to its kind, or names a target with no value given.
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        controls = wells.controls
        count = len(self.well_names)
        control_modes = broadcast_or_match(
            self.control_mode, count=count, field_name="control_mode"
        )
        values = broadcast_or_match(self.value, count=count, field_name="value")
        per_well_mode = isinstance(control_modes, (list, tuple, np.ndarray))
        per_well_value = isinstance(values, (list, tuple, np.ndarray))

        resolved_modes: list[ProducerControlMode | InjectorControlMode] = []
        by_target_field: dict[str, list[tuple[Integer, Number]]] = {}
        for index, (well_name, well_row) in enumerate(
            zip(self.well_names, well_rows, strict=True)
        ):
            one_mode = control_modes[index] if per_well_mode else control_modes
            is_injector = controls.well_kinds[well_row] == WellKind.INJECTOR
            mode_map = INJECTOR_CONTROL_MODE_MAP if is_injector else PRODUCER_CONTROL_MODE_MAP
            deck_mode = one_mode.value
            try:
                resolved_modes.append(mode_map[deck_mode])
            except KeyError:
                raise ValidationError(
                    f"Control mode {one_mode.name} doesn't apply to "
                    f"{'an injector' if is_injector else 'a producer'} ({well_name!r})."
                ) from None

            target_field = WELTARG_TARGET_FIELD[deck_mode]
            if target_field is None:
                continue  # GROUP: only the mode changes
            one_value = values[index] if per_well_value else values
            if one_value is None:
                raise ValidationError(
                    f"Well {well_name!r} names control mode {one_mode.name}, which needs "
                    "a value, but none was given."
                )
            by_target_field.setdefault(target_field, []).append((well_row, one_value))

        controls.set_control_mode(well_row=well_rows, mode=resolved_modes)
        for target_field, pairs in by_target_field.items():
            setter = TARGET_SETTERS[target_field]
            setter(controls, [row for row, _ in pairs], [value for _, value in pairs])
        return model


def as_list(
    value: BulkValue | tuple[BulkValue, ...], *, count: Integer, field_name: str
) -> list[BulkValue]:
    """
    Turns a bulk field into an explicit per-target list.

    :param value: A single value (broadcast to every target) or a
        sequence already aligned with the targets.
    :param count: How many targets this field must align with.
    :param field_name: The field's own name, for a validation message.
    :returns: A list of length `count`.
    :raises ValidationError: If `value` is a sequence of the wrong length.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != count:
            raise ValidationError(
                f"{field_name!r} has {len(value)} value(s) but {count} target(s) were expected."
            )
        return list(value)
    return [value] * count


def find_limit_rows(
    *,
    limits: CompiledLimits,
    well_names: typing.Sequence[str],
    well_rows: IntArray[OneDimension],
    kinds: typing.Sequence[LimitKind],
    quantities: typing.Sequence[RateQuantity | EconomicQuantity | None],
) -> IntArray[OneDimension]:
    """
    Finds every `(well, kind, quantity)` target's limit row, all at once.

    :param limits: The compiled limits table to search.
    :param well_names: Each target's well name, for a validation message.
    :param well_rows: Each target's well row, positionally matched with `well_names`.
    :param kinds: Each target's limit kind, positionally matched with `well_names`.
    :param quantities: Each target's quantity, positionally matched with `well_names`.
    :returns: Each target's own limit row, in the same order.
    :raises ValidationError: If any target had no matching limit from the start of the run.
    """
    rows: list[Integer] = []
    missing: list[str] = []
    for well_name, well_row, kind, quantity in zip(
        well_names, well_rows, kinds, quantities, strict=True
    ):
        row = limits.find_limit_row(well_row=well_row, kind=kind, quantity=quantity)
        if row is None:
            missing.append(
                f"{well_name!r} ({kind!r}" + (f", {quantity!r})" if quantity is not None else ")")
            )
        else:
            rows.append(row)
    if missing:
        raise ValidationError(
            "No matching limit from the start of the run for: "
            f"{', '.join(missing)}. A limit cannot be added mid-run."
        )
    return np.asarray(rows, dtype=np.intp)  # type: ignore[return-value]


def apply_limit_values(
    *,
    limits: CompiledLimits,
    rows: IntArray[OneDimension],
    count: Integer,
    min_values: Number | tuple[Number, ...] | None,
    max_values: Number | tuple[Number, ...] | None,
    workover_actions: WorkoverAction | tuple[WorkoverAction, ...] | None,
    end_runs: Boolean | tuple[Boolean, ...] | None,
) -> None:
    """
    Overwrites every value field given, in place, on `rows`.

    Every field given is validated (via `broadcast_or_match`) before any
    of them are written, so a bad field length never leaves `rows`
    partially patched.

    :param limits: The compiled limits table to patch.
    :param rows: The limit rows to write to.
    :param count: How many targets `rows` covers, for length validation
        on any field given as a sequence.
    :param min_values: The new floor(s), if given.
    :param max_values: The new ceiling(s), if given.
    :param workover_actions: The new workover action(s), if given.
    :param end_runs: The new end-run flag(s), if given.
    """
    resolved_min = (
        None
        if min_values is None
        else broadcast_or_match(min_values, count=count, field_name="min_values")
    )
    resolved_max = (
        None
        if max_values is None
        else broadcast_or_match(max_values, count=count, field_name="max_values")
    )
    resolved_workover = (
        None
        if workover_actions is None
        else broadcast_or_match(workover_actions, count=count, field_name="workover_actions")
    )
    resolved_end_run = (
        None
        if end_runs is None
        else broadcast_or_match(end_runs, count=count, field_name="end_runs")
    )

    if resolved_min is not None:
        limits.set_min_value(row=rows, value=as_number_or_array(resolved_min))
    if resolved_max is not None:
        limits.set_max_value(row=rows, value=as_number_or_array(resolved_max))
    if resolved_workover is not None:
        limits.set_workover_action(row=rows, action=resolved_workover)
    if resolved_end_run is not None:
        limits.set_end_run(row=rows, end_run=resolved_end_run)


@attrs.frozen(kw_only=True, slots=True)
class SetLimits(Action["CompiledBlackOilModel"]):
    """
    Changes existing limits across several `(well, limit)` targets in one bulk write. A deck
    `WECON` reissue loads as one of these per quantity, and so does the BHP or THP limit
    carried by a `WCONPROD` or `WCONINJE` record.

    `mode=ONE_TO_ONE` pairs `well_names[i]` with `kinds[i]` (and
    `quantities[i]`, when given) position by position: each index names
    one distinct target, so `kinds` (and any given `quantities`) must
    have the same length as `well_names`.

    `mode=ALL` treats `kinds` (and `quantities`) as a set of limit
    specs, each applied to every well in `well_names`, a full cross
    product. `min_values`/`max_values`/`workover_actions`/`end_runs`
    then vary by spec, not by well: give one value per entry in
    `kinds`, or a single value to apply to every spec.

    In either mode, a value field left `None` is not touched on any target.
    """

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to act on."""

    kinds: tuple[LimitKind, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """Which limit(s) to update. Paired with `well_names` per `mode`."""

    quantities: (
        RateQuantity | EconomicQuantity | tuple[RateQuantity | EconomicQuantity | None, ...] | None
    ) = None
    """Which quantity each entry in `kinds` targets. A single value
    broadcasts to every entry; ignored for `BHP`/`THP`."""

    min_values: Number | tuple[Number, ...] | None = None
    """The new floor(s), if given."""

    max_values: Number | tuple[Number, ...] | None = None
    """The new ceiling(s), if given."""

    workover_actions: WorkoverAction | tuple[WorkoverAction, ...] | None = None
    """The new workover action(s), if given. Only meaningful for `kind=ECONOMIC`."""

    end_runs: Boolean | tuple[Boolean, ...] | None = None
    """The new end-run flag(s), if given. Only meaningful for `kind=ECONOMIC`."""

    mode: ApplicationMode = ApplicationMode.ONE_TO_ONE
    """How `well_names` pairs with `kinds`/`quantities`."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Resolves every target's limit row, then overwrites every value field given.

        Every target is resolved before any value is written, so a
        missing limit or a bad field length never leaves some targets
        patched and others not.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with every target's limit patched.
        :raises ValidationError: If `kinds`/`quantities` don't line up
            with `well_names` per `mode`, or a target has no matching
            limit from the start of the run.
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        limits = wells.controls.limits
        if self.mode == ApplicationMode.ONE_TO_ONE:
            self.apply_one_to_one(limits=limits, well_rows=well_rows)
        else:
            self.apply_all(limits=limits, well_rows=well_rows)
        return model

    def apply_one_to_one(
        self, *, limits: CompiledLimits, well_rows: IntArray[OneDimension]
    ) -> None:
        """
        Resolves and patches one `(well, kind, quantity)` target per `well_names` index.

        :param limits: The compiled limits table to patch.
        :param well_rows: Every well's own row, same order as `well_names`.
        :raises ValidationError: If `kinds` isn't the same length as
            `well_names`, or a target had no matching limit from the start of the run.
        """
        count = len(self.well_names)
        if len(self.kinds) != count:
            raise ValidationError(
                f"{type(self).__name__} in `ONE_TO_ONE` mode needs exactly {count} `kinds` (one per "
                f"well in `well_names`); got {len(self.kinds)}."
            )
        quantities = as_list(self.quantities, count=count, field_name="quantities")
        rows = find_limit_rows(
            limits=limits,
            well_names=self.well_names,
            well_rows=well_rows,
            kinds=self.kinds,
            quantities=quantities,
        )
        apply_limit_values(
            limits=limits,
            rows=rows,
            count=count,
            min_values=self.min_values,
            max_values=self.max_values,
            workover_actions=self.workover_actions,
            end_runs=self.end_runs,
        )

    def apply_all(self, *, limits: CompiledLimits, well_rows: IntArray[OneDimension]) -> None:
        """
        Resolves and patches every entry in `kinds` against every well in `well_names`.

        Every spec's rows are resolved for every well before any value
        is written, so a missing limit for one spec never leaves an
        earlier spec's wells patched while a later spec's are not.

        :param limits: The compiled limits table to patch.
        :param well_rows: Every well's own row, same order as `well_names`.
        :raises ValidationError: If a target had no matching limit from the start of the run.
        """
        count = len(self.kinds)
        quantities = as_list(self.quantities, count=count, field_name="quantities")
        min_values = (
            None
            if self.min_values is None
            else as_list(self.min_values, count=count, field_name="min_values")
        )
        max_values = (
            None
            if self.max_values is None
            else as_list(self.max_values, count=count, field_name="max_values")
        )
        workover_actions = (
            None
            if self.workover_actions is None
            else as_list(self.workover_actions, count=count, field_name="workover_actions")
        )
        end_runs = (
            None
            if self.end_runs is None
            else as_list(self.end_runs, count=count, field_name="end_runs")
        )

        rows_per_spec = [
            find_limit_rows(
                limits=limits,
                well_names=self.well_names,
                well_rows=well_rows,
                kinds=[kind] * len(well_rows),
                quantities=[quantity] * len(well_rows),
            )
            for kind, quantity in zip(self.kinds, quantities, strict=True)
        ]
        for index, rows in enumerate(rows_per_spec):
            apply_limit_values(
                limits=limits,
                rows=rows,
                count=len(well_rows),
                min_values=None if min_values is None else min_values[index],
                max_values=None if max_values is None else max_values[index],
                workover_actions=None if workover_actions is None else workover_actions[index],
                end_runs=None if end_runs is None else end_runs[index],
            )


@attrs.frozen(kw_only=True, slots=True)
class SetGroupLimit(Action["CompiledBlackOilModel"]):
    """
    Updates one of a group's existing `GECON` limit rows in place, the
    group-level counterpart of `SetLimits`. A `GECON` reissue loads as one of these.

    Only an already-compiled limit row matching `quantity` can be patched.
    """

    group_name: str
    """The group to act on."""

    quantity: EconomicQuantity
    """Which quantity to update. A group's own limits are always `ECONOMIC`."""

    min_value: Number | None = None
    """The new floor, if given."""

    max_value: Number | None = None
    """The new ceiling, if given."""

    workover_action: WorkoverAction | None = None
    """The new workover action, if given."""

    end_run: Boolean | None = None
    """The new end-run flag, if given."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Overwrites every field given, in place, on the group's matching limit row.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the group's limit patched.
        :raises ValidationError: If the group had no matching limit from the start of the run.
        """
        group_row, wells = resolve_group(model=model, group_name=self.group_name)
        limits: CompiledGroupLimits = wells.group_controls.limits  # type: ignore[union-attr]
        row = limits.find_limit_row(group_row=group_row, quantity=self.quantity)
        if row is None:
            raise ValidationError(
                f"Group {self.group_name!r} has no `ECONOMIC` limit for {self.quantity!r} from "
                "the start of the run, and a limit cannot be added mid-run."
            )
        if self.min_value is not None:
            limits.set_min_value(row=row, value=self.min_value)
        if self.max_value is not None:
            limits.set_max_value(row=row, value=self.max_value)
        if self.workover_action is not None:
            limits.set_workover_action(row=row, action=self.workover_action)
        if self.end_run is not None:
            limits.set_end_run(row=row, end_run=self.end_run)
        return model


RATE_ARRAYS: dict[tuple[RateQuantity, Boolean], str] = {
    (RateQuantity.OIL, False): "oil_rates",
    (RateQuantity.WATER, False): "water_rates",
    (RateQuantity.GAS, False): "gas_rates",
    (RateQuantity.OIL, True): "surface_oil_rates",
    (RateQuantity.WATER, True): "surface_water_rates",
    (RateQuantity.GAS, True): "surface_gas_rates",
}
"""Maps a `(quantity, surface)` pair to the `WellsWorkspace` array it reads from."""


def get_well_rate_arrays(
    *, model: "CompiledBlackOilModel", context: ScheduleContext, well_name: str, caller: str
) -> tuple[Integer, "WellsWorkspace"]:
    """
    The well's row and the workspace's well arrays, for an event that reads solved rates.

    :param model: The model being scheduled against.
    :param context: The current moment's context.
    :param well_name: The well to read.
    :param caller: The reading event's name, for the error message.
    :returns: `(well_row, workspace.wells)`.
    :raises ValidationError: If `context.extra` has no `SimulationWorkspace` under `"workspace"`.
    """
    from bores.simulation.workspace import SimulationWorkspace

    workspace = context.extra.get("workspace")
    if not isinstance(workspace, SimulationWorkspace):
        raise ValidationError(
            f"{caller} needs key 'workspace' in `context.extra`, with a `SimulationWorkspace` "
            f"value, to read the rates of well {well_name!r}, but got {workspace!r}."
        )
    well_row, _ = resolve_well(model=model, well_name=well_name)
    return well_row, workspace.wells


@attrs.frozen(kw_only=True, slots=True)
class RateThreshold(ThresholdEvent["CompiledBlackOilModel"]):
    """Fires when a well's phase rate, from the latest solve, satisfies `op` against `threshold`."""

    well_name: str
    """The well to watch."""

    quantity: RateQuantity
    """Which rate to watch: `OIL`, `WATER`, `GAS`, or `LIQUID` (oil plus water)."""

    surface: Boolean = False
    """Surface-condition rate if `True`, reservoir-condition otherwise."""

    def get_value(self, *, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads the well's own current rate from the workspace.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: The well's current rate for `quantity`/`surface`.
        :raises ValidationError: If `context.extra` has no `workspace`, or if `quantity`/`surface`
            isn't supported.
        """
        well_row, wells = get_well_rate_arrays(
            model=model, context=context, well_name=self.well_name, caller=type(self).__name__
        )
        if self.quantity == RateQuantity.LIQUID:
            names = [
                RATE_ARRAYS[RateQuantity.OIL, self.surface],
                RATE_ARRAYS[RateQuantity.WATER, self.surface],
            ]
            return sum(getattr(wells, name)[well_row] for name in names)
        array_name = RATE_ARRAYS.get((self.quantity, self.surface))
        if array_name is None:
            raise ValidationError(
                f"{type(self).__name__} doesn't support `quantity={self.quantity!r}`, "
                f"surface={self.surface!r}."
            )
        return getattr(wells, array_name)[well_row]


@attrs.frozen(kw_only=True, slots=True)
class WaterCutThreshold(ThresholdEvent["CompiledBlackOilModel"]):
    """Fires when a well's surface water cut, from the latest solve, satisfies `op` against `threshold`."""

    well_name: str
    """The well to watch."""

    def get_value(self, *, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads the well's water cut: surface water rate over surface liquid rate.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: The water cut, `0` when the well produces no liquid.
        """
        well_row, wells = get_well_rate_arrays(
            model=model,
            context=context,
            well_name=self.well_name,
            caller=type(self).__name__,
        )
        oil_rate = wells.surface_oil_rates[well_row]
        water_rate = wells.surface_water_rates[well_row]
        liquid_rate = oil_rate + water_rate
        return water_rate / liquid_rate if liquid_rate > 0.0 else 0.0


@attrs.frozen(kw_only=True, slots=True)
class GasOilRatioThreshold(ThresholdEvent["CompiledBlackOilModel"]):
    """Fires when a well's surface gas-oil ratio, from the latest solve, satisfies `op` against `threshold`."""

    well_name: str
    """The well to watch."""

    def get_value(self, *, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads the well's gas-oil ratio: surface gas rate over surface oil rate.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: The gas-oil ratio, infinite for gas with no oil, `0` for neither.
        """
        well_row, wells = get_well_rate_arrays(
            model=model,
            context=context,
            well_name=self.well_name,
            caller=type(self).__name__,
        )
        oil_rate = wells.surface_oil_rates[well_row]
        gas_rate = wells.surface_gas_rates[well_row]
        if oil_rate > 0.0:
            return gas_rate / oil_rate
        return float("inf") if gas_rate > 0.0 else 0.0


@attrs.frozen(kw_only=True, slots=True)
class SetWellLift(Action["CompiledBlackOilModel"]):
    """
    Changes a well's VFP table, artificial lift quantity (ALQ) and efficiency factor.

    Each setting that is `None` is left as it is. A deck `WLIFT` record loads as one of these,
    and so do the lift items of a `WCONPROD` or `WCONINJE` reissue.
    """

    well_name: str
    """The well to act on."""

    vfp_table: Integer | None = None
    """The VFP table number to switch to, or `0` for no table. `None` keeps the current table."""

    artificial_lift_quantity: Number | None = None
    """
    The ALQ to operate at, in the units of the table's ALQ axis. Producers only. `None`
    keeps the current value.
    """

    efficiency_factor: Number | None = None
    """The well's new efficiency factor, in `(0, 1]`. `None` keeps the current value."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Applies the lift settings that were given.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the well's lift settings patched.
        :raises ValidationError: If an ALQ is given for an injector, or a value is invalid.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        controls = wells.controls
        if (
            self.artificial_lift_quantity is not None
            and controls.well_kinds[well_row] == WellKind.INJECTOR
        ):
            raise ValidationError(
                f"{type(self).__name__}: well {self.well_name!r} is an injector, which has no "
                "artificial lift quantity."
            )
        if self.artificial_lift_quantity is not None and not (
            np.isfinite(self.artificial_lift_quantity) and self.artificial_lift_quantity >= 0.0
        ):
            raise ValidationError(
                f"{type(self).__name__}: the artificial lift quantity must be finite and "
                f"non-negative; got {self.artificial_lift_quantity}."
            )
        if self.vfp_table is not None and self.vfp_table < 0:
            raise ValidationError(
                f"{type(self).__name__}: the VFP table number must be `0` (no table) or "
                f"positive; got {self.vfp_table}."
            )
        if self.efficiency_factor is not None and not (0.0 < self.efficiency_factor <= 1.0):
            raise ValidationError(
                f"{type(self).__name__}: the efficiency factor must be in (0, 1]; got "
                f"{self.efficiency_factor}."
            )

        if self.vfp_table is not None:
            controls.set_vfp_table_number(
                well_row=well_row, value=UNSET_INT if self.vfp_table == 0 else self.vfp_table
            )
        if self.artificial_lift_quantity is not None:
            controls.set_artificial_lift_quantity(
                well_row=well_row, value=self.artificial_lift_quantity
            )
        if self.efficiency_factor is not None:
            controls.set_efficiency_factor(well_row=well_row, value=self.efficiency_factor)
        return model


def iter_actions(action: Action) -> typing.Iterator[Action]:
    """
    :param action: An action, which may be a `RunSequence` of others.
    :returns: Every action it consists of, looking inside any `RunSequence`.
    """
    if isinstance(action, RunSequence):
        for inner in action.actions:
            yield from iter_actions(inner)
    else:
        yield action


def check_vfp_references(*, well_system: "WellSystem", schedule: "Schedule | None" = None) -> None:
    """
    Checks, before a run, that every VFP table number a well uses exists.

    A well uses a table number from its own control and from every `SetWellLift` in the
    schedule that names it. Wells whose tubing performance is not a collection of VFP tables
    are skipped, since they never look a table up. A table number is looked up among the
    producer or the injector tables, according to the well's kind.

    :param well_system: The wells, their controls and their tubing performance.
    :param schedule: The run's schedule, if it has one.
    :raises ValidationError: Listing every table number that is not found, with where it comes from.
    """
    references: dict[str, list[tuple[Integer, str]]] = {name: [] for name in well_system.wells}
    for name in references:
        control = well_system.well_controls.get(name)
        number = getattr(control, "vfp_table", None)
        if number:
            references[name].append((number, "its initial control"))

    if schedule is not None:
        for item in schedule.items:
            for action in iter_actions(item.action):
                if (
                    isinstance(action, SetWellLift)
                    and action.vfp_table
                    and action.well_name in references
                ):
                    references[action.well_name].append((
                        action.vfp_table,
                        f"schedule item {item.name!r}",
                    ))

    missing = []
    for name, numbers in references.items():
        tubing_performance = well_system.get_tubing_performance(name)
        if not isinstance(tubing_performance, VFPTables):
            continue

        well = well_system.wells[name]
        tables = tubing_performance.injectors if well.is_injector else tubing_performance.producers
        kind = "injector" if well.is_injector else "producer"
        missing.extend(
            f"well {name!r} ({kind}) uses VFP table {number} from {source}"
            for number, source in numbers
            if number not in tables
        )
    if missing:
        raise ValidationError(
            "These VFP tables are not defined for the wells that use them: "
            + "; ".join(missing)
            + "."
        )
