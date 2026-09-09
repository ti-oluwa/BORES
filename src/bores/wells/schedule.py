"""Well-specific `Event`/`Action` implementations, built on `bores.schedule`."""

import attrs

from bores.blackoil.compile import CompiledBlackOilModel
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.schedule.base import ScheduleContext, SerializableAction, action_type
from bores.schedule.events import ThresholdEvent
from bores.types import Number
from bores.wells.base import CompletionStatus
from bores.wells.compile import (
    INJECTOR_MODE_TAG,
    PRODUCER_MODE_TAG,
    CompiledPerforations,
    CompiledWellSystem,
    WellKind,
)
from bores.wells.controls import RateQuantity
from bores.wells.deck import INJECTOR_CONTROL_MODE_MAP, PRODUCER_CONTROL_MODE_MAP
from bores.wells.resolution.compile import CompiledWellResolution

__all__ = ["ConnectionFactorMultiplierAction", "RateEvent", "WellOpenAction", "WellTargetAction"]


def get_well_system(
    model: CompiledBlackOilModel, well_name: str
) -> tuple[int, CompiledWellSystem]:
    """
    :param model: The model to look `well_name` up in.
    :param well_name: The well to find.
    :returns: `(well_row, model.wells)`.
    :raises ValidationError: If `model.wells` is `None`.
    """
    if model.wells is None:
        raise ValidationError(f"{model!r} has no compiled wells.")
    return model.wells.well_row(well_name), model.wells


def get_connection_rows(
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
    all_rows = perforations.connection_rows(well_row)
    if i == 0 and j == 0 and k1 == 0 and k2 == 0:
        return all_rows

    if grid is None or grid.dimensions is None:
        raise ValidationError("A grid with dimensions is required to target specific connections.")
    dims = grid.dimensions
    target_cells = {dims.flat_index(i - 1, j - 1, k - 1) for k in range(k1, k2 + 1)}
    return [row for row in all_rows if perforations.get_cell_index(row) in target_cells]


@action_type
@attrs.frozen(kw_only=True, slots=True)
class WellOpenAction(SerializableAction[CompiledBlackOilModel]):
    """`WELOPEN`: opens or shuts a whole well or specific connections."""

    well_name: str
    status: CompletionStatus
    i: int = 0
    j: int = 0
    k1: int = 0
    k2: int = 0

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
        well_row, wells = get_well_system(model, self.well_name)
        grid = model.reservoir.grid if self.i or self.j or self.k1 or self.k2 else None
        rows = get_connection_rows(
            wells.perforations, well_row, self.i, self.j, self.k1, self.k2, grid
        )
        for row in rows:
            wells.perforations.set_completion_status(row, self.status)
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class WellTargetAction(SerializableAction[CompiledBlackOilModel]):
    """`WELTARG`: changes a well's control mode and the one target value it names."""

    well_name: str
    control_mode: str
    value: float | None = None

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
        well_row, wells = get_well_system(model, self.well_name)
        controls = wells.controls
        is_injector = controls.well_kinds[well_row] == WellKind.INJECTOR
        mode_map = INJECTOR_CONTROL_MODE_MAP if is_injector else PRODUCER_CONTROL_MODE_MAP
        tag_map = INJECTOR_MODE_TAG if is_injector else PRODUCER_MODE_TAG

        try:
            new_mode = mode_map[self.control_mode]
        except KeyError:
            raise ValidationError(
                f"`WELTARG` control mode {self.control_mode!r} doesn't apply to "
                f"{'an injector' if is_injector else 'a producer'} ({self.well_name!r})."
            ) from None
        controls.control_modes[well_row] = tag_map[new_mode]  # type: ignore

        setter = TARGET_SETTERS.get(self.control_mode)
        if setter is None:
            return model  # GRUP: only the mode changes
        if self.value is None:
            raise ValidationError(
                f"`WELTARG` on well {self.well_name!r} names control mode "
                f"{self.control_mode!r}, which needs a value, but none was given."
            )
        setter(controls, well_row, self.value)
        return model


TARGET_SETTERS = {
    "ORAT": lambda c, r, v: c.set_target_rate(r, v),
    "WRAT": lambda c, r, v: c.set_target_rate(r, v),
    "GRAT": lambda c, r, v: c.set_target_rate(r, v),
    "LRAT": lambda c, r, v: c.set_target_rate(r, v),
    "RESV": lambda c, r, v: c.set_target_rate(r, v),
    "RATE": lambda c, r, v: c.set_target_rate(r, v),
    "BHP": lambda c, r, v: c.set_target_bhp(r, v),
    "THP": lambda c, r, v: c.set_target_thp(r, v),
}
"""Which `CompiledWellControls` setter a `WELTARG` control-mode string writes with."""


@action_type
@attrs.frozen(kw_only=True, slots=True)
class ConnectionFactorMultiplierAction(SerializableAction[CompiledBlackOilModel]):
    """`WPIMULT`: multiplies existing connection factors in place."""

    well_name: str
    multiplier: float
    i: int = 0
    j: int = 0
    k1: int = 0
    k2: int = 0

    def __call__(
        self, model: CompiledBlackOilModel, context: ScheduleContext
    ) -> CompiledBlackOilModel:
        well_row, wells = get_well_system(model, self.well_name)
        grid = model.reservoir.grid if self.i or self.j or self.k1 or self.k2 else None
        rows = get_connection_rows(
            wells.perforations, well_row, self.i, self.j, self.k1, self.k2, grid
        )
        for row in rows:
            wells.perforations.multiply_well_index(row, self.multiplier)
        return model


RATE_ARRAY = {
    (RateQuantity.OIL, False): "oil_rates",
    (RateQuantity.WATER, False): "water_rates",
    (RateQuantity.GAS, False): "gas_rates",
    (RateQuantity.OIL, True): "surface_oil_rates",
    (RateQuantity.WATER, True): "surface_water_rates",
    (RateQuantity.GAS, True): "surface_gas_rates",
}
"""Which `CompiledWellResolution` array a `(quantity, surface)` pair reads from."""


@attrs.frozen(kw_only=True, slots=True)
class RateEvent(ThresholdEvent[CompiledBlackOilModel]):
    """Fires when a well's phase rate, from the latest solve, crosses `threshold`."""

    well_name: str
    quantity: RateQuantity
    surface: bool = False
    """Surface-condition rate if `True`, reservoir-condition otherwise."""

    def get_value(self, model: CompiledBlackOilModel, context: ScheduleContext) -> Number:
        if not isinstance(context.state, CompiledWellResolution):
            raise ValidationError("RateEvent needs context.state to be a CompiledWellResolution.")

        well_row, _ = get_well_system(model, self.well_name)
        key = (self.quantity, self.surface)
        array_name = RATE_ARRAY.get(key)
        if array_name is None:
            raise ValidationError(
                f"RateEvent doesn't support quantity={self.quantity!r}, surface={self.surface!r}."
            )
        return float(getattr(context.state, array_name)[well_row])
