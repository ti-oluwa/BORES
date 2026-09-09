"""Well-specific `Event`/`Action` implementations, built on `bores.schedule`."""

import attrs

from bores.blackoil.compile import CompiledBlackOilModel
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.schedule.base import ScheduleContext, SerializableAction, action_type
from bores.schedule.events import ThresholdEvent, event_type
from bores.types import FluidPhase, Number
from bores.wells.base import CompletionStatus, WellStatus
from bores.wells.compile import CompiledPerforations, CompiledWellSystem, WellKind
from bores.wells.controls import (
    InjectorControlMode,
    ProducerControlMode,
    RateQuantity,
    WorkoverAction,
)
from bores.wells.deck import (
    INJECTOR_CONTROL_MODE_MAP,
    PRODUCER_CONTROL_MODE_MAP,
    WELTARG_TARGET_FIELD,
)
from bores.wells.resolution.compile import CompiledWellResolution

__all__ = [
    "RATE_ARRAYS",
    "TARGET_SETTERS",
    "ActivateCompletion",
    "MultiplyConnectionFactor",
    "OpenWell",
    "RateThreshold",
    "SetEconomicLimit",
    "SetWellControl",
    "SetWellTarget",
    "get_matching_connection_rows",
    "resolve_well",
]


def resolve_well(
    *, model: CompiledBlackOilModel, well_name: str
) -> tuple[int, CompiledWellSystem]:
    """
    Finds a well's row and its compiled well system within a model.

    :param model: The model to look `well_name` up in.
    :param well_name: The well to find.
    :returns: `(well_row, model.wells)`.
    :raises ValidationError: If `model.wells` is `None`.
    """
    if model.wells is None:
        raise ValidationError(f"{model!r} has no compiled wells.")
    return model.wells.well_row(name=well_name), model.wells


def get_matching_connection_rows(
    *,
    perforations: CompiledPerforations,
    well_row: int,
    i: int,
    j: int,
    k1: int,
    k2: int,
    grid: Grid | None,
) -> range | list[int]:
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
    :returns: The matching row indices.
    :raises ValidationError: If connections are targeted but `grid` wasn't given.
    """
    all_rows = perforations.connection_rows(well_row=well_row)
    if i == 0 and j == 0 and k1 == 0 and k2 == 0:
        return all_rows

    if grid is None or grid.dimensions is None:
        raise ValidationError("A grid with dimensions is required to target specific connections.")
    dims = grid.dimensions
    target_cells = {dims.flat_index(i=i - 1, j=j - 1, k=k - 1) for k in range(k1, k2 + 1)}
    return [row for row in all_rows if perforations.get_cell_index(row=row) in target_cells]


@action_type
@attrs.frozen(kw_only=True, slots=True)
class OpenWell(SerializableAction[CompiledBlackOilModel]):
    """`WELOPEN`: opens or shuts a whole well or specific connections."""

    well_name: str
    """The well to act on."""

    status: CompletionStatus
    """The new open/shut status."""

    i: int = 0
    """1-based deck index, or `0` for whole-well."""

    j: int = 0
    """1-based deck index, or `0`."""

    k1: int = 0
    """1-based deck index, or `0`."""

    k2: int = 0
    """1-based deck index, or `0`."""

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
        """
        Sets the targeted connection(s)' open/shut status.

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
        for row in rows:
            wells.perforations.set_completion_status(row=row, status=self.status)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class MultiplyConnectionFactor(SerializableAction[CompiledBlackOilModel]):
    """`WPIMULT`: multiplies existing connection factors in place."""

    well_name: str
    """The well to act on."""

    multiplier: float
    """The multiplier to apply."""

    i: int = 0
    """1-based deck index, or `0` for whole-well."""

    j: int = 0
    """1-based deck index, or `0`."""

    k1: int = 0
    """1-based deck index, or `0`."""

    k2: int = 0
    """1-based deck index, or `0`."""

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
        """
        Multiplies the targeted connection(s)' connection factor.

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
        for row in rows:
            wells.perforations.multiply_well_index(row=row, factor=self.multiplier)
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
class SetWellTarget(SerializableAction[CompiledBlackOilModel]):
    """`WELTARG`: changes a well's control mode and the one target value it names."""

    well_name: str
    """The well to act on."""

    control_mode: str
    """Deck item 2's literal string (`ORAT`, `BHP`, `GRUP`, and so on), pre-translation."""

    value: float | None = None
    """Deck item 3. `None` only valid when `control_mode` is `GRUP` (takes no value)."""

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
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

        try:
            new_mode = mode_map[self.control_mode]
        except KeyError:
            raise ValidationError(
                f"WELTARG control mode {self.control_mode!r} doesn't apply to "
                f"{'an injector' if is_injector else 'a producer'} ({self.well_name!r})."
            ) from None
        controls.set_control_mode(well_row=well_row, mode=new_mode)

        target_field = WELTARG_TARGET_FIELD[self.control_mode]
        if target_field is None:
            return model  # GRUP: only the mode changes
        if self.value is None:
            raise ValidationError(
                f"WELTARG on well {self.well_name!r} names control mode "
                f"{self.control_mode!r}, which needs a value, but none was given."
            )
        setter = TARGET_SETTERS[target_field]
        setter(controls, well_row, self.value)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetWellControl(SerializableAction[CompiledBlackOilModel]):
    """`WCONPROD`/`WCONINJE`: redefines a well's control mode and targets in one go."""

    well_name: str
    """The well to act on."""

    mode: ProducerControlMode | InjectorControlMode
    """The new control mode. Must match the well's own kind."""

    target_rate: float | None = None
    """The new target rate, if given."""

    target_bhp: float | None = None
    """The new target BHP, if given."""

    target_thp: float | None = None
    """The new target THP, if given."""

    injected_phase: FluidPhase | None = None
    """The new injected phase, if given. Only meaningful on an injector."""

    efficiency_factor: float | None = None
    """The new efficiency factor, if given."""

    guide_rate: float | None = None
    """The new guide rate, if given."""

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
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
class ActivateCompletion(SerializableAction[CompiledBlackOilModel]):
    """
    `COMPDAT`: activates a workover completion already sitting `PENDING`
    in the compiled arrays since compile time.
    """

    well_name: str
    """The well to act on."""

    i: int = 0
    """1-based deck index, or `0` for whole-well."""

    j: int = 0
    """1-based deck index, or `0`."""

    k1: int = 0
    """1-based deck index, or `0`."""

    k2: int = 0
    """1-based deck index, or `0`."""

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
        """
        Sets the targeted connection(s)' schedule status to `ACTIVE`.

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
        for row in rows:
            wells.perforations.set_schedule_status(row=row, status=WellStatus.ACTIVE)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class SetEconomicLimit(SerializableAction[CompiledBlackOilModel]):
    """
    `WECON`: updates a well's existing economic limit in place.

    Only an already-compiled `ECONOMIC`-kind limit row can be patched.
    `CompiledLimits`' CSR table has no slack to grow a brand new row for a
    well with no economic limit at compile time; that needs Step 7's
    unsolved CSR-growth problem, not this action.
    """

    well_name: str
    """The well to act on."""

    min_value: float | None = None
    """The new floor, if given."""

    max_value: float | None = None
    """The new ceiling, if given."""

    workover_action: WorkoverAction | None = None
    """The new workover action, if given."""

    end_run: bool | None = None
    """The new end-run flag, if given."""

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
        """
        Overwrites every field given, in place, on the well's economic limit.

        :param model: The model to change.
        :param context: The current moment's context. Unused.
        :returns: `model`, with the well's economic limit patched.
        :raises ValidationError: If the well has no economic limit at compile time.
        """
        well_row, wells = resolve_well(model=model, well_name=self.well_name)
        limits = wells.controls.limits
        row = limits.find_economic_limit_row(well_row=well_row)
        if row is None:
            raise ValidationError(
                f"Well {self.well_name!r} has no economic limit at compile time. "
                "Adding one mid-schedule needs CompiledLimits' CSR table to grow, "
                "which is not supported yet."
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


RATE_ARRAYS = {
    (RateQuantity.OIL, False): "oil_rates",
    (RateQuantity.WATER, False): "water_rates",
    (RateQuantity.GAS, False): "gas_rates",
    (RateQuantity.OIL, True): "surface_oil_rates",
    (RateQuantity.WATER, True): "surface_water_rates",
    (RateQuantity.GAS, True): "surface_gas_rates",
}
"""Maps a `(quantity, surface)` pair to the `CompiledWellResolution` array it reads from."""


@event_type
@attrs.frozen(kw_only=True, slots=True)
class RateThreshold(ThresholdEvent[CompiledBlackOilModel]):
    """Fires when a well's phase rate, from the latest solve, crosses `threshold`."""

    well_name: str
    """The well to watch."""

    quantity: RateQuantity
    """Which phase rate to watch. Only `OIL`, `WATER`, `GAS` are supported."""

    surface: bool = False
    """Surface-condition rate if `True`, reservoir-condition otherwise."""

    def get_value(self, *, model: CompiledBlackOilModel, context: ScheduleContext) -> Number:
        """
        Reads the well's own current rate from `context.state`.

        :param model: The model being scheduled against.
        :param context: The current moment's context. `context.state`
            must be a `CompiledWellResolution`.
        :returns: The well's current rate for `quantity`/`surface`.
        :raises ValidationError: If `context.state` isn't a
            `CompiledWellResolution`, or `quantity`/`surface` isn't supported.
        """
        if not isinstance(context.state, CompiledWellResolution):
            raise ValidationError(
                f"{type(self).__name__} needs context.state to be a CompiledWellResolution."
            )
        well_row, _ = resolve_well(model=model, well_name=self.well_name)
        array_name = RATE_ARRAYS.get((self.quantity, self.surface))
        if array_name is None:
            raise ValidationError(
                f"{type(self).__name__} doesn't support quantity={self.quantity!r}, surface={self.surface!r}."
            )
        return float(getattr(context.state, array_name)[well_row])
