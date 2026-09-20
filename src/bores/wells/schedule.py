"""Well-specific `Event`/`Action` implementations, built on `bores.schedule`."""

import enum
import typing

import attrs
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.schedule.base import ScheduleContext, SerializableAction, action_type
from bores.schedule.events import ThresholdEvent, event_type
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
    CompiledGroupLimits,
    CompiledLimits,
    CompiledPerforations,
    CompiledWellSystem,
    LimitKind,
    WellKind,
)
from bores.wells.controls import (
    EconomicQuantity,
    InjectorControlMode,
    ProducerControlMode,
    RateQuantity,
    WellTargetMode,
    WorkoverAction,
)
from bores.wells.mappings import (
    INJECTOR_CONTROL_MODE_MAP,
    PRODUCER_CONTROL_MODE_MAP,
    WELTARG_TARGET_FIELD,
)

if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel


__all__ = [
    "RATE_ARRAYS",
    "TARGET_SETTERS",
    "ActivateCompletion",
    "ActivateCompletions",
    "ActivateWell",
    "ActivateWells",
    "ApplicationMode",
    "MultiplyConnectionFactor",
    "MultiplyConnectionFactors",
    "OpenWell",
    "OpenWells",
    "RateThreshold",
    "SetGroupLimit",
    "SetLimit",
    "SetLimits",
    "SetWellControl",
    "SetWellControls",
    "SetWellTarget",
    "SetWellTargets",
    "as_number_or_array",
    "broadcast_or_match",
    "expand_values",
    "get_many_matching_connection_rows",
    "get_matching_connection_rows",
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


def get_matching_connection_rows(
    *,
    perforations: CompiledPerforations,
    well_row: Integer,
    i: Integer,
    j: Integer,
    k1: Integer,
    k2: Integer,
    grid: Grid | None = None,
) -> IntArray[OneDimension]:
    """
    Resolves which of a well's connection rows a deck targeting tuple selects.

    `(0, 0, 0, 0)` means every connection. A targeted `(i, j, k1, k2)`
    needs `grid` to translate 1-based deck indices to flat cell indices.

    :param perforations: `CompiledPerforations` for the whole system.
    :param well_row: The well's row.
    :param i: 1-based deck index, or `0` for whole-well.
    :param j: 1-based deck index, or `0`.
    :param k1: 1-based deck index, or `0`.
    :param k2: 1-based deck index, or `0`.
    :param grid: Required only when `(i, j, k1, k2)` targets specific connections.
    :returns: The matching row indices, as an array (bulk-accessor-ready).
    :raises ValidationError: If connections are targeted but `grid` wasn't given.
    """
    all_rows = perforations.connection_rows(well_row=well_row)
    if i == 0 and j == 0 and k1 == 0 and k2 == 0:
        return np.arange(all_rows.start, all_rows.stop)

    if grid is None or grid.dimensions is None:
        raise ValidationError("A grid with dimensions is required to target specific connections.")
    dims = grid.dimensions
    target_cells = {dims.flat_index(i=i - 1, j=j - 1, k=k - 1) for k in range(k1, k2 + 1)}
    cell_indices = perforations.get_cell_index(row=np.arange(all_rows.start, all_rows.stop))
    mask = np.isin(cell_indices, list(target_cells))  # type: ignore[arg-type]
    return typing.cast(IntArray[OneDimension], np.arange(all_rows.start, all_rows.stop)[mask])


def get_many_matching_connection_rows(
    *,
    perforations: CompiledPerforations,
    well_rows: IntArray[OneDimension],
    i: Integer,
    j: Integer,
    k1: Integer,
    k2: Integer,
    grid: Grid | None,
) -> tuple[IntArray[OneDimension], list[int]]:
    """
    Resolves matching connection rows across several wells at once.

    :param perforations: `CompiledPerforations` for the whole system.
    :param well_rows: Every well's own row.
    :param i: 1-based deck index, or `0` for whole-well.
    :param j: 1-based deck index, or `0`.
    :param k1: 1-based deck index, or `0`.
    :param k2: 1-based deck index, or `0`.
    :param grid: Required only when `(i, j, k1, k2)` targets specific connections.
    :returns: Every well's matching rows, concatenated in `well_rows`'
        order, and, in the same order, how many rows each well
        contributed (to later align a per-well value with the
        concatenated rows via `expand_values`).
    """
    rows_per_well = [
        get_matching_connection_rows(
            perforations=perforations, well_row=well_row, i=i, j=j, k1=k1, k2=k2, grid=grid
        )
        for well_row in well_rows
    ]
    counts = [len(rows) for rows in rows_per_well]
    if not rows_per_well:
        return np.empty(0, dtype=np.intp), counts  # type: ignore[return-value]
    return np.concatenate(rows_per_well), counts  # type: ignore[return-value]


@action_type
@attrs.frozen(kw_only=True, slots=True)
class OpenWell(SerializableAction["CompiledBlackOilModel"]):
    """
    Opens or shuts a whole well, or specific connections on it. Build
    this directly for a manual shut-in exactly as freely as any other
    action; it also happens to be what a deck `WELOPEN` record becomes
    when `load_schedule` reads one.
    """

    __type__: typing.ClassVar[str] = "open_well"

    well_name: str
    """The well to act on."""

    status: CompletionStatus
    """The new open/shut status."""

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
        Sets the targeted connection(s)' open/shut status, in one bulk write.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the targeted connections patched.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        grid = model.reservoir.grid if (self.i or self.j or self.k1 or self.k2) else None
        rows = get_matching_connection_rows(
            perforations=wells.perforations,
            well_row=well_row,
            i=self.i,
            j=self.j,
            k1=self.k1,
            k2=self.k2,
            grid=grid,
        )
        wells.perforations.set_completion_status(row=rows, status=self.status)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class MultiplyConnectionFactor(SerializableAction["CompiledBlackOilModel"]):
    """
    Multiplies a well's existing connection factor(s) in place - a
    whole-well or per-connection productivity adjustment, constructed
    directly or, equally, produced from a deck `WPIMULT` record by
    `load_schedule`.
    """

    __type__: typing.ClassVar[str] = "multiply_connection_factor"

    well_name: str
    """The well to act on."""

    multiplier: float
    """The multiplier to apply."""

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
        Multiplies the targeted connection(s)' connection factor, in one bulk write.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the targeted connections patched.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        grid = model.reservoir.grid if (self.i or self.j or self.k1 or self.k2) else None
        rows = get_matching_connection_rows(
            perforations=wells.perforations,
            well_row=well_row,
            i=self.i,
            j=self.j,
            k1=self.k1,
            k2=self.k2,
            grid=grid,
        )
        wells.perforations.multiply_well_index(row=rows, factor=self.multiplier)
        return model


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


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetWellTarget(SerializableAction["CompiledBlackOilModel"]):
    """
    Changes a well's control mode, and the one target value that mode
    names, in one step. Build this directly for a manual control change
    exactly as freely as any other action; it also happens to be what a
    deck `WELTARG` record becomes when `load_schedule` reads one.
    """

    __type__: typing.ClassVar[str] = "set_well_target"

    well_name: str
    """The well to act on."""

    control_mode: WellTargetMode
    """The mode to switch to. Must apply to this well's own kind (a
    producer or an injector) - see `WellTargetMode`."""

    value: Number | None = None
    """The new value for whichever field `control_mode` names (a rate,
    BHP, or THP). `None` is only valid when `control_mode` is `GROUP`,
    which takes no value."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Sets the well's control mode, and its named target value if it has one.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the well's control patched.
        :raises ValidationError: If `control_mode` doesn't apply to this
            well's kind, or names a target with no `value` given.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        controls = wells.controls
        is_injector = controls.well_kinds[well_row] == WellKind.INJECTOR
        mode_map = INJECTOR_CONTROL_MODE_MAP if is_injector else PRODUCER_CONTROL_MODE_MAP
        deck_mode = self.control_mode.value

        try:
            new_mode = mode_map[deck_mode]
        except KeyError:
            raise ValidationError(
                f"{type(self).__name__} control mode {self.control_mode.name} doesn't apply to "
                f"{'an injector' if is_injector else 'a producer'} ({self.well_name!r})."
            ) from None
        controls.set_control_mode(well_row=well_row, mode=new_mode)

        target_field = WELTARG_TARGET_FIELD[deck_mode]
        if target_field is None:
            return model  # GROUP: only the mode changes
        if self.value is None:
            raise ValidationError(
                f"{type(self).__name__} on well {self.well_name!r} names control mode "
                f"{self.control_mode.name}, which needs a value, but none was given."
            )
        setter = TARGET_SETTERS[target_field]
        setter(controls, well_row, self.value)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetWellControl(SerializableAction["CompiledBlackOilModel"]):
    """
    Redefines a well's control mode and targets in one go. Just as
    valid to build directly for a manual control change as it is to
    load from a deck - `load_schedule` produces one from each
    `WCONPROD`/`WCONINJE` record it reads.
    """

    __type__: typing.ClassVar[str] = "set_well_control"

    well_name: str
    """The well to act on."""

    mode: ProducerControlMode | InjectorControlMode
    """The new control mode. Must match the well's own kind."""

    target_rate: Number | None = None
    """The new target rate, if given."""

    target_bhp: Number | None = None
    """The new target BHP, if given."""

    target_thp: Number | None = None
    """The new target THP, if given."""

    injected_phase: FluidPhase | None = None
    """The new injected phase, if given. Only meaningful on an injector."""

    efficiency_factor: Number | None = None
    """The new efficiency factor, if given."""

    guide_rate: Number | None = None
    """The new guide rate, if given."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Overwrites every field given, in place, on the well's control.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the well's control patched.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        controls = wells.controls
        controls.set_control_mode(well_row=well_row, mode=self.mode)
        if self.target_rate is not None:
            controls.set_target_rate(well_row=well_row, value=self.target_rate)
        if self.target_bhp is not None:
            controls.set_target_bhp(well_row=well_row, value=self.target_bhp)
        if self.target_thp is not None:
            controls.set_target_thp(well_row=well_row, value=self.target_thp)
        if self.injected_phase is not None:
            controls.set_injected_phase(well_row=well_row, phase=self.injected_phase)
        if self.efficiency_factor is not None:
            controls.set_efficiency_factor(well_row=well_row, value=self.efficiency_factor)
        if self.guide_rate is not None:
            controls.set_guide_rate(well_row=well_row, value=self.guide_rate)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class ActivateCompletion(SerializableAction["CompiledBlackOilModel"]):
    """
    Activates a well's connection(s), flipping their schedule status
    from `PENDING` to `ACTIVE`. Every connection is compiled up front,
    at compile time, regardless of when it actually starts flowing -
    this is what brings one online, whether built by hand for a
    scripted/manual schedule or produced from a deck `COMPDAT` record
    by `load_schedule`.
    """

    __type__: typing.ClassVar[str] = "activate_completion"

    well_name: str
    """The well to act on."""

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
        Sets the targeted connection(s)' schedule status to `ACTIVE`, in one bulk write.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the targeted connections activated.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        grid = model.reservoir.grid if (self.i or self.j or self.k1 or self.k2) else None
        rows = get_matching_connection_rows(
            perforations=wells.perforations,
            well_row=well_row,
            i=self.i,
            j=self.j,
            k1=self.k1,
            k2=self.k2,
            grid=grid,
        )
        wells.perforations.set_schedule_status(row=rows, status=WellStatus.ACTIVE)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class ActivateWell(SerializableAction["CompiledBlackOilModel"]):
    """
    Activates a well, flipping its schedule status from `PENDING` to
    `ACTIVE`. Every well is compiled up front, the moment it's first
    mentioned anywhere in the deck, regardless of when it actually
    starts - this is what brings one online, whether built by hand or
    produced from a deck `WELSPECS` record by `load_schedule`.

    This only flips the well-level status; each of its connections
    still activates on its own schedule via `ActivateCompletion`, which
    `load_schedule` emits separately for a real `COMPDAT` record.
    """

    __type__: typing.ClassVar[str] = "activate_well"

    well_name: str
    """The well to activate."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Sets the well's own schedule status to `ACTIVE`.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the well activated.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        wells.set_schedule_status(well_row=well_row, status=WellStatus.ACTIVE)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetLimit(SerializableAction["CompiledBlackOilModel"]):
    """
    Updates one of a well's existing limit rows in place - build this
    directly for a manual limit change exactly as freely as any other
    action. `load_schedule` also produces one from a `WECON` reissue
    (`kind=ECONOMIC`), or from the implicit `BHPLimit` a `WCONPROD`/
    `WCONINJE` record's own `bhp` item carries when its control mode
    isn't `BHP` (`kind=BHP`).

    Only an already-compiled limit row matching `kind` (and `quantity`,
    for `RATE`/`ECONOMIC`) can be patched. `CompiledLimits`' CSR table
    has no slack to grow a brand new row for a well with no such limit at
    compile time; that needs Step 7's unsolved CSR-growth problem, not
    this action. A single `WECON` record can define several economic
    limits at once (a max water cut and a max GOR, say) -
    `load_schedule` emits one `SetLimit` per quantity in that case, not
    one per record.
    """

    __type__: typing.ClassVar[str] = "set_limit"

    well_name: str
    """The well to act on."""

    kind: LimitKind
    """Which kind of limit to update."""

    quantity: RateQuantity | EconomicQuantity | None = None
    """Which quantity to update, for `kind=RATE` or `kind=ECONOMIC`. Ignored for `BHP`/`THP`."""

    min_value: Number | None = None
    """The new floor, if given."""

    max_value: Number | None = None
    """The new ceiling, if given."""

    workover_action: WorkoverAction | None = None
    """The new workover action, if given. Only meaningful for `kind=ECONOMIC`."""

    end_run: Boolean | None = None
    """The new end-run flag, if given. Only meaningful for `kind=ECONOMIC`."""

    def __call__(
        self, model: "CompiledBlackOilModel", context: ScheduleContext
    ) -> "CompiledBlackOilModel":
        """
        Overwrites every field given, in place, on the well's matching limit row.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the well's limit patched.
        :raises ValidationError: If the well has no matching limit at compile time.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        limits = wells.controls.limits
        row = limits.find_limit_row(well_row=well_row, kind=self.kind, quantity=self.quantity)
        if row is None:
            raise ValidationError(
                f"Well {self.well_name!r} has no {self.kind!r} limit"
                f"{f' for {self.quantity!r}' if self.quantity is not None else ''} at compile "
                "time. Adding one mid-schedule needs `CompiledLimits`' CSR table to grow, "
                "which is not supported."
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


@action_type
@attrs.frozen(kw_only=True, slots=True)
class ActivateWells(SerializableAction["CompiledBlackOilModel"]):
    """`ActivateWell`, applied to several wells at once."""

    __type__: typing.ClassVar[str] = "bulk_activate_well"

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


@action_type
@attrs.frozen(kw_only=True, slots=True)
class OpenWells(SerializableAction["CompiledBlackOilModel"]):
    """
    `OpenWell`, applied to several wells at once, all against the same
    targeted connection(s).
    """

    __type__: typing.ClassVar[str] = "bulk_open_well"

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
        )
        if isinstance(status, (list, tuple, np.ndarray)):
            status = expand_values(status, counts=counts)
        wells.perforations.set_completion_status(row=rows, status=status)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class MultiplyConnectionFactors(SerializableAction["CompiledBlackOilModel"]):
    """
    `MultiplyConnectionFactor`, applied to several wells at once, all
    against the same targeted connection(s).
    """

    __type__: typing.ClassVar[str] = "bulk_multiply_connection_factor"

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
        )
        if isinstance(multiplier, (list, tuple, np.ndarray)):
            multiplier = expand_values(multiplier, counts=counts)
        wells.perforations.multiply_well_index(row=rows, factor=as_number_or_array(multiplier))
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class ActivateCompletions(SerializableAction["CompiledBlackOilModel"]):
    """
    `ActivateCompletion`, applied to several wells at once, all
    against the same targeted connection(s).
    """

    __type__: typing.ClassVar[str] = "bulk_activate_completion"

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
        grid = model.reservoir.grid if (self.i or self.j or self.k1 or self.k2) else None
        rows, _ = get_many_matching_connection_rows(
            perforations=wells.perforations,
            well_rows=well_rows,
            i=self.i,
            j=self.j,
            k1=self.k1,
            k2=self.k2,
            grid=grid,
        )
        wells.perforations.set_schedule_status(row=rows, status=WellStatus.ACTIVE)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetWellControls(SerializableAction["CompiledBlackOilModel"]):
    """`SetWellControl`, applied to several wells at once."""

    __type__: typing.ClassVar[str] = "bulk_set_well_control"

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
        """
        well_rows, wells = resolve_wells(model=model, well_names=self.well_names)
        controls = wells.controls
        n = len(self.well_names)
        controls.set_control_mode(
            well_row=well_rows, mode=broadcast_or_match(self.mode, count=n, field_name="mode")
        )
        if self.target_rate is not None:
            controls.set_target_rate(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.target_rate, count=n, field_name="target_rate")
                ),
            )
        if self.target_bhp is not None:
            controls.set_target_bhp(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.target_bhp, count=n, field_name="target_bhp")
                ),
            )
        if self.target_thp is not None:
            controls.set_target_thp(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.target_thp, count=n, field_name="target_thp")
                ),
            )
        if self.injected_phase is not None:
            controls.set_injected_phase(
                well_row=well_rows,
                phase=broadcast_or_match(
                    self.injected_phase, count=n, field_name="injected_phase"
                ),
            )
        if self.efficiency_factor is not None:
            controls.set_efficiency_factor(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(
                        self.efficiency_factor, count=n, field_name="efficiency_factor"
                    )
                ),
            )
        if self.guide_rate is not None:
            controls.set_guide_rate(
                well_row=well_rows,
                value=as_number_or_array(
                    broadcast_or_match(self.guide_rate, count=n, field_name="guide_rate")
                ),
            )
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetWellTargets(SerializableAction["CompiledBlackOilModel"]):
    """
    `SetWellTarget`, applied to several wells at once. Each well's own
    kind (producer or injector) still picks which of
    `ProducerControlMode`/`InjectorControlMode` `control_mode` resolves
    to, so a mixed set of producers and injectors is fine even when
    `control_mode` is broadcast to all of them.
    """

    __type__: typing.ClassVar[str] = "bulk_set_well_target"

    well_names: tuple[str, ...] = attrs.field(
        converter=tuple, validator=attrs.validators.min_len(1)
    )
    """The wells to act on."""

    control_mode: WellTargetMode | tuple[WellTargetMode, ...]
    """The mode(s) to switch to - see `WellTargetMode`. A single value
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
        n = len(self.well_names)
        control_modes = broadcast_or_match(self.control_mode, count=n, field_name="control_mode")
        values = broadcast_or_match(self.value, count=n, field_name="value")
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
                    f"{type(self).__name__} control mode {one_mode.name} doesn't apply to "
                    f"{'an injector' if is_injector else 'a producer'} ({well_name!r})."
                ) from None

            target_field = WELTARG_TARGET_FIELD[deck_mode]
            if target_field is None:
                continue  # GROUP: only the mode changes
            one_value = values[index] if per_well_value else values
            if one_value is None:
                raise ValidationError(
                    f"{type(self).__name__} on well {well_name!r} names control mode "
                    f"{one_mode.name}, which needs a value, but none was given."
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
    action_name: str,
) -> IntArray[OneDimension]:
    """
    Finds every `(well, kind, quantity)` target's limit row, all at once.

    :param limits: The compiled limits table to search.
    :param well_names: Each target's well name, for a validation message.
    :param well_rows: Each target's well row, positionally matched with `well_names`.
    :param kinds: Each target's limit kind, positionally matched with `well_names`.
    :param quantities: Each target's quantity, positionally matched with `well_names`.
    :param action_name: The calling action's class name, for a validation message.
    :returns: Each target's own limit row, in the same order.
    :raises ValidationError: If any target has no matching limit at compile time.
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
            f"{action_name} found no matching limit at compile time for: {', '.join(missing)}. "
            "Adding one mid-schedule needs `CompiledLimits`' CSR table to grow, which is not "
            "supported."
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


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetLimits(SerializableAction["CompiledBlackOilModel"]):
    """
    `SetLimit`, applied across several `(well, limit)` targets in one bulk write.

    `mode=ONE_TO_ONE` pairs `well_names[i]` with `kinds[i]` (and
    `quantities[i]`, when given) position by position: each index names
    one distinct target, so `kinds` (and any given `quantities`) must
    have the same length as `well_names`.

    `mode=ALL` treats `kinds` (and `quantities`) as a set of limit
    specs, each applied to every well in `well_names`, a full cross
    product. `min_values`/`max_values`/`workover_actions`/`end_runs`
    then vary by spec, not by well: give one value per entry in
    `kinds`, or a single value to apply to every spec.

    In either mode, a value field left `None` is not touched on any
    target, the same as the single-well `SetLimit`.
    """

    __type__: typing.ClassVar[str] = "bulk_set_limit"

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
            limit at compile time.
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
            `well_names`, or a target has no matching limit at compile time.
        """
        n = len(self.well_names)
        if len(self.kinds) != n:
            raise ValidationError(
                f"{type(self).__name__} in `ONE_TO_ONE` mode needs exactly {n} `kinds` (one per "
                f"well in `well_names`); got {len(self.kinds)}."
            )
        quantities = as_list(self.quantities, count=n, field_name="quantities")
        rows = find_limit_rows(
            limits=limits,
            well_names=self.well_names,
            well_rows=well_rows,
            kinds=self.kinds,
            quantities=quantities,
            action_name=type(self).__name__,
        )
        apply_limit_values(
            limits=limits,
            rows=rows,
            count=n,
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
        :raises ValidationError: If a target has no matching limit at compile time.
        """
        m = len(self.kinds)
        quantities = as_list(self.quantities, count=m, field_name="quantities")
        min_values = (
            None
            if self.min_values is None
            else as_list(self.min_values, count=m, field_name="min_values")
        )
        max_values = (
            None
            if self.max_values is None
            else as_list(self.max_values, count=m, field_name="max_values")
        )
        workover_actions = (
            None
            if self.workover_actions is None
            else as_list(self.workover_actions, count=m, field_name="workover_actions")
        )
        end_runs = (
            None
            if self.end_runs is None
            else as_list(self.end_runs, count=m, field_name="end_runs")
        )

        rows_per_spec = [
            find_limit_rows(
                limits=limits,
                well_names=self.well_names,
                well_rows=well_rows,
                kinds=[kind] * len(well_rows),
                quantities=[quantity] * len(well_rows),
                action_name=type(self).__name__,
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


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetGroupLimit(SerializableAction["CompiledBlackOilModel"]):
    """
    Updates one of a group's existing `GECON` limit rows in place, the
    group-level counterpart of `SetLimit`. A `GECON` reissue.

    Only an already-compiled limit row matching `quantity` can be patched.
    """

    __type__: typing.ClassVar[str] = "set_group_limit"

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
        :raises ValidationError: If the group has no matching limit at compile time.
        """
        group_row, wells = resolve_group(model=model, group_name=self.group_name)
        limits: CompiledGroupLimits = wells.group_controls.limits  # type: ignore[union-attr]
        row = limits.find_limit_row(group_row=group_row, quantity=self.quantity)
        if row is None:
            raise ValidationError(
                f"Group {self.group_name!r} has no `ECONOMIC` limit for {self.quantity!r} at "
                "compile time. Adding one mid-schedule needs `CompiledGroupLimits`' CSR table "
                "to grow, which is not supported."
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


@event_type
@attrs.frozen(kw_only=True, slots=True)
class RateThreshold(ThresholdEvent["CompiledBlackOilModel"]):
    """Fires when a well's phase rate, from the latest solve, crosses `threshold`."""

    __type__: typing.ClassVar[str] = "rate_threshold"

    well_name: str
    """The well to watch."""

    quantity: RateQuantity
    """Which phase rate to watch. Only `OIL`, `WATER`, `GAS` are supported."""

    surface: Boolean = False
    """Surface-condition rate if `True`, reservoir-condition otherwise."""

    def get_value(self, *, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads the well's own current rate from `context.state`.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: The well's current rate for `quantity`/`surface`.
        :raises ValidationError: If `context.extra` has no `workspace`,
        or if `quantity`/`surface` isn't supported.
        """
        from bores.simulation.workspace import SimulationWorkspace

        workspace = context.extra.get("workspace")
        if not isinstance(workspace, SimulationWorkspace):
            raise ValidationError(
                f"{type(self).__name__} needs key 'workspace' in `context.extra`, with a `SimulationWorkspace` value, to read "
                f"{self.quantity!r} {self.surface!r} rate for well {self.well_name!r}, but got {workspace!r}."
            )

        well_row, _ = resolve_well(model=model, well_name=self.well_name)
        array_name = RATE_ARRAYS.get((self.quantity, self.surface))
        if array_name is None:
            raise ValidationError(
                f"{type(self).__name__} doesn't support `quantity={self.quantity!r}`, "
                f"surface={self.surface!r}."
            )
        return getattr(workspace.wells, array_name)[well_row]
