"""Compiled (Structure-of-Arrays) well data for the reservoir solver hot path."""

import enum
import math
import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.types import (
    FluidPhase,
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    Orientation,
    UnitSystem,
)
from bores.wells.base import (
    AnyPerforation,
    CompletionStatus,
    Well,
    Wells,
    WellStatus,
    WellType,
)
from bores.wells.controls import (
    BHPLimit,
    EconomicLimit,
    EconomicQuantity,
    InjectorControl,
    InjectorControlMode,
    Limit,
    ProducerControl,
    ProducerControlMode,
    RateLimit,
    RateQuantity,
    THPLimit,
    WellControls,
    WorkoverAction,
)
from bores.wells.groups import (
    GroupControls,
    GroupInjectorControlMode,
    GroupProducerControlMode,
    WellGroups,
)
from bores.wells.indices.perforations import PerforationIndex
from bores.wells.indices.wells import build_wells_indices

__all__ = [
    "CompiledGroupControls",
    "CompiledLimits",
    "CompiledPerforations",
    "CompiledWellControls",
    "CompiledWellSystem",
    "GroupKind",
    "LimitKind",
    "WellKind",
    "_compile_limits",
    "compile_group_controls",
    "compile_perforations",
    "compile_well_controls",
    "compile_well_system",
]


UNSET_INT = -1
"""Sentinel for an int tag column entry that doesn't apply to that row."""


class WellKind(enum.IntEnum):
    """Tag value for `CompiledWellSystem.well_kinds`/`CompiledWellControls.well_kinds`."""

    PRODUCER = 0
    INJECTOR = 1


class LimitKind(enum.IntEnum):
    """Tag value for `CompiledLimits.kinds`."""

    BHP = 0
    RATE = 1
    THP = 2
    ECONOMIC = 3


class GroupKind(enum.IntEnum):
    """Tag value for `CompiledGroupControls.group_kinds`."""

    PRODUCER = 0
    INJECTOR = 1


class ProducerControlModeTag(enum.IntEnum):
    """Tag value for `CompiledWellControls.control_modes` on a `WellKind.PRODUCER` row."""

    OIL_RATE = 0
    WATER_RATE = 1
    GAS_RATE = 2
    LIQUID_RATE = 3
    RESERVOIR_VOLUME_RATE = 4
    BHP = 5
    THP = 6
    GROUP = 7
    UNSET = 8


class InjectorControlModeTag(enum.IntEnum):
    """
    Tag value for `CompiledWellControls.control_modes` on a
    `WellKind.INJECTOR` row. Check `well_kinds` before interpreting a
    `control_modes` entry.

    This is a separate tag space from `ProducerControlModeTag`, even though both start at `0`.
    """

    RATE = 0
    RESERVOIR_VOLUME_RATE = 1
    BHP = 2
    THP = 3
    GROUP = 4
    UNSET = 5


class GroupProducerControlModeTag(enum.IntEnum):
    """Tag value for `CompiledGroupControls.control_modes` on a `GroupKind.PRODUCER` row."""

    OIL_RATE = 0
    WATER_RATE = 1
    GAS_RATE = 2
    LIQUID_RATE = 3
    RESERVOIR_VOLUME_RATE = 4
    FIELD = 5
    NONE = 6


class GroupInjectorControlModeTag(enum.IntEnum):
    """Tag value for `CompiledGroupControls.control_modes` on a `GroupKind.INJECTOR` row."""

    RATE = 0
    RESERVOIR_VOLUME_RATE = 1
    VOIDAGE_REPLACEMENT = 2
    REINJECTION = 3
    FIELD = 4


class RateQuantityTag(enum.IntEnum):
    """Tag value for `CompiledLimits.quantities` on a `RATE` row."""

    OIL = 0
    WATER = 1
    GAS = 2
    LIQUID = 3
    RESERVOIR = 4


class EconomicQuantityTag(enum.IntEnum):
    """Tag value for `CompiledLimits.quantities` on an `ECONOMIC` row."""

    WATER_CUT = 0
    GOR = 1
    WATER_GAS_RATIO = 2
    OIL_RATE = 3
    GAS_RATE = 4


class WorkoverActionTag(enum.IntEnum):
    """Tag value for `CompiledLimits.workover_actions` on an `ECONOMIC` row."""

    WELL = 0
    PLUG = 1
    CON = 2
    PLUS_CON = 3


class FluidPhaseTag(enum.IntEnum):
    """Tag value for `CompiledWellControls.injected_phases`/`CompiledGroupControls.injected_phases`."""

    OIL = 0
    WATER = 1
    GAS = 2


PRODUCER_MODE_TAG = {
    ProducerControlMode.OIL_RATE: ProducerControlModeTag.OIL_RATE,
    ProducerControlMode.WATER_RATE: ProducerControlModeTag.WATER_RATE,
    ProducerControlMode.GAS_RATE: ProducerControlModeTag.GAS_RATE,
    ProducerControlMode.LIQUID_RATE: ProducerControlModeTag.LIQUID_RATE,
    ProducerControlMode.RESERVOIR_VOLUME_RATE: ProducerControlModeTag.RESERVOIR_VOLUME_RATE,
    ProducerControlMode.BHP: ProducerControlModeTag.BHP,
    ProducerControlMode.THP: ProducerControlModeTag.THP,
    ProducerControlMode.GROUP: ProducerControlModeTag.GROUP,
    ProducerControlMode.UNSET: ProducerControlModeTag.UNSET,
}
INJECTOR_MODE_TAG = {
    InjectorControlMode.RATE: InjectorControlModeTag.RATE,
    InjectorControlMode.RESERVOIR_VOLUME_RATE: InjectorControlModeTag.RESERVOIR_VOLUME_RATE,
    InjectorControlMode.BHP: InjectorControlModeTag.BHP,
    InjectorControlMode.THP: InjectorControlModeTag.THP,
    InjectorControlMode.GROUP: InjectorControlModeTag.GROUP,
    InjectorControlMode.UNSET: InjectorControlModeTag.UNSET,
}
GROUP_PRODUCER_MODE_TAG = {
    GroupProducerControlMode.OIL_RATE: GroupProducerControlModeTag.OIL_RATE,
    GroupProducerControlMode.WATER_RATE: GroupProducerControlModeTag.WATER_RATE,
    GroupProducerControlMode.GAS_RATE: GroupProducerControlModeTag.GAS_RATE,
    GroupProducerControlMode.LIQUID_RATE: GroupProducerControlModeTag.LIQUID_RATE,
    GroupProducerControlMode.RESERVOIR_VOLUME_RATE: GroupProducerControlModeTag.RESERVOIR_VOLUME_RATE,
    GroupProducerControlMode.FIELD: GroupProducerControlModeTag.FIELD,
    GroupProducerControlMode.NONE: GroupProducerControlModeTag.NONE,
}
GROUP_INJECTOR_MODE_TAG = {
    GroupInjectorControlMode.RATE: GroupInjectorControlModeTag.RATE,
    GroupInjectorControlMode.RESERVOIR_VOLUME_RATE: GroupInjectorControlModeTag.RESERVOIR_VOLUME_RATE,
    GroupInjectorControlMode.VOIDAGE_REPLACEMENT: GroupInjectorControlModeTag.VOIDAGE_REPLACEMENT,
    GroupInjectorControlMode.REINJECTION: GroupInjectorControlModeTag.REINJECTION,
    GroupInjectorControlMode.FIELD: GroupInjectorControlModeTag.FIELD,
}
GROUP_TO_PRODUCER_MODE_TAG = {
    GroupProducerControlModeTag.OIL_RATE: ProducerControlModeTag.OIL_RATE,
    GroupProducerControlModeTag.WATER_RATE: ProducerControlModeTag.WATER_RATE,
    GroupProducerControlModeTag.GAS_RATE: ProducerControlModeTag.GAS_RATE,
    GroupProducerControlModeTag.LIQUID_RATE: ProducerControlModeTag.LIQUID_RATE,
    GroupProducerControlModeTag.RESERVOIR_VOLUME_RATE: ProducerControlModeTag.RESERVOIR_VOLUME_RATE,
}
"""
Maps a `GroupProducerControlModeTag` to the concrete `ProducerControlModeTag`
a `GRUP`-mode member well switches to once allocated a share of it.

`FIELD`/`NONE` are deliberately absent because neither has a directly allocatable per-well target.
"""

GROUP_TO_INJECTOR_MODE_TAG = {
    GroupInjectorControlModeTag.RATE: InjectorControlModeTag.RATE,
    GroupInjectorControlModeTag.RESERVOIR_VOLUME_RATE: InjectorControlModeTag.RESERVOIR_VOLUME_RATE,
}
"""
Injector analogue of `GROUP_TO_PRODUCER_MODE_TAG`. `VOIDAGE_REPLACEMENT`/
`REINJECTION`/`FIELD` deliberately absent, same reasoning.
"""

RATE_QUANTITY_TAG = {
    RateQuantity.OIL: RateQuantityTag.OIL,
    RateQuantity.WATER: RateQuantityTag.WATER,
    RateQuantity.GAS: RateQuantityTag.GAS,
    RateQuantity.LIQUID: RateQuantityTag.LIQUID,
    RateQuantity.RESERVOIR: RateQuantityTag.RESERVOIR,
}
ECONOMIC_QUANTITY_TAG = {
    EconomicQuantity.WATER_CUT: EconomicQuantityTag.WATER_CUT,
    EconomicQuantity.GOR: EconomicQuantityTag.GOR,
    EconomicQuantity.WATER_GAS_RATIO: EconomicQuantityTag.WATER_GAS_RATIO,
    EconomicQuantity.OIL_RATE: EconomicQuantityTag.OIL_RATE,
    EconomicQuantity.GAS_RATE: EconomicQuantityTag.GAS_RATE,
}
WORKOVER_ACTION_TAG = {
    WorkoverAction.WELL: WorkoverActionTag.WELL,
    WorkoverAction.PLUG: WorkoverActionTag.PLUG,
    WorkoverAction.CON: WorkoverActionTag.CON,
    WorkoverAction.PLUS_CON: WorkoverActionTag.PLUS_CON,
}
FLUID_PHASE_TAG = {
    FluidPhase.OIL: FluidPhaseTag.OIL,
    FluidPhase.WATER: FluidPhaseTag.WATER,
    FluidPhase.GAS: FluidPhaseTag.GAS,
}
PRODUCER_MODE_FROM_TAG = {tag: mode for mode, tag in PRODUCER_MODE_TAG.items()}
INJECTOR_MODE_FROM_TAG = {tag: mode for mode, tag in INJECTOR_MODE_TAG.items()}
FLUID_PHASE_FROM_TAG = {tag: phase for phase, tag in FLUID_PHASE_TAG.items()}
RATE_QUANTITY_FROM_TAG = {tag: quantity for quantity, tag in RATE_QUANTITY_TAG.items()}
ECONOMIC_QUANTITY_FROM_TAG = {tag: quantity for quantity, tag in ECONOMIC_QUANTITY_TAG.items()}
WORKOVER_ACTION_FROM_TAG = {tag: action for action, tag in WORKOVER_ACTION_TAG.items()}
"""
Inverses of the `*_TAG` maps above, for accessor methods to translate a
stored tag back into the rich enum a caller actually wants to see.
"""


def none_if_nan(value: float) -> float | None:
    """
    Returns `value`, or `None` if it's `NaN`.

    :param value: A possibly-`NaN` float read from a compiled array.
    :returns: `value`, or `None` if it's `NaN`.
    """
    return None if math.isnan(value) else value


class CompiledPerforations(typing.NamedTuple):
    """
    Every well's perforations, flattened row-per-connection and
    CSR-indexed by well. Includes every perforation regardless of status -
    a shut or not-yet-scheduled connection is a tagged row here, not an
    absent one, so a later status change never needs to re-resolve geometry.
    """

    well_offsets: IntArray[OneDimension]
    """Shape `(n_wells + 1,)`. Well `w`'s rows are `well_offsets[w]:well_offsets[w + 1]`."""

    cell_indices: IntArray[OneDimension]
    """Shape `(n_rows,)`. The grid cell each connection resolves to."""

    perforation_indices: IntArray[OneDimension]
    """
    Shape `(n_rows,)`. Position of the rich `Perforation`/`MDPerforation`
    this row came from, within that well's own `Well.perforations`. One
    rich perforation can resolve to several rows (a trajectory crossing
    several grid cells), so this is not the same as the row's own
    position - it lets a row be traced back to its source without relying
    on row position, which decompiling needs since `CompiledPerforations`
    keeps no other reference to the rich perforations it was built from.
    """

    well_indices: NumberArray[OneDimension]
    """Shape `(n_rows,)`. Each connection's connection factor."""

    wellbore_radii: NumberArray[OneDimension]
    """Shape `(n_rows,)`."""

    skins: NumberArray[OneDimension]
    """Shape `(n_rows,)`."""

    partial_penetration_fractions: NumberArray[OneDimension]
    """Shape `(n_rows,)`."""

    representative_depths: NumberArray[OneDimension]
    """Shape `(n_rows,)`. For the hydrostatic term in wellbore pressure-drop."""

    inclinations_from_vertical: NumberArray[OneDimension]
    """Shape `(n_rows,)`, radians. For `wells.hydraulics`."""

    completion_statuses: IntArray[OneDimension]
    """Shape `(n_rows,)`. `1` for `CompletionStatus.OPEN`, `0` for `SHUT`."""

    schedule_statuses: IntArray[OneDimension]
    """
    Shape `(n_rows,)`. `1` for `WellStatus.ACTIVE`, `0` for `PENDING`.
    A row with either this or `completion_statuses` at `0` is skipped by
    every solver kernel.
    """

    saturation_regions: IntArray[OneDimension]
    """Shape `(n_rows,)`. Per-connection SATNUM override; `-1` means use the cell's own region."""

    def connection_rows(self, *, well_row: Integer) -> range:
        """
        A well's own row range, without the caller needing to know about `well_offsets`.

        :param well_row: The well's row in `CompiledWellSystem.names`.
        :returns: `range(start, end)` over this well's rows.
        """
        return range(int(self.well_offsets[well_row]), int(self.well_offsets[well_row + 1]))

    def get_cell_index(self, *, row: Integer) -> int:
        """
        :param row: The connection row to read.
        :returns: The grid cell this connection resolves to.
        """
        return int(self.cell_indices[row])

    def get_well_index(self, *, row: Integer) -> float:
        """
        :param row: The connection row to read.
        :returns: This connection's own connection factor (well index).
        """
        return float(self.well_indices[row])

    def set_well_index(self, *, row: Integer, value: Number) -> None:
        """
        Overwrites a connection's connection factor in place.

        :param row: The connection row to write.
        :param value: The new connection factor.
        """
        self.well_indices[row] = value

    def multiply_well_index(self, *, row: Integer, factor: Number) -> None:
        """
        Multiplies a connection's existing connection factor in place (a `WPIMULT` reissue).

        :param row: The connection row to write.
        :param factor: The multiplier to apply.
        """
        self.well_indices[row] *= factor

    def get_completion_status(self, *, row: Integer) -> CompletionStatus:
        """
        :param row: The connection row to read.
        :returns: This connection's own open/shut status.
        """
        return CompletionStatus.OPEN if self.completion_statuses[row] else CompletionStatus.SHUT

    def set_completion_status(self, *, row: Integer, status: CompletionStatus) -> None:
        """
        Sets a connection's open/shut status in place (a `WELOPEN` event).

        :param row: The connection row to write.
        :param status: The new open/shut status.
        """
        self.completion_statuses[row] = 1 if status == CompletionStatus.OPEN else 0

    def get_schedule_status(self, *, row: Integer) -> WellStatus:
        """
        :param row: The connection row to read.
        :returns: This connection's own active/pending status.
        """
        return WellStatus.ACTIVE if self.schedule_statuses[row] else WellStatus.PENDING

    def set_schedule_status(self, *, row: Integer, status: WellStatus) -> None:
        """
        Sets a connection's active/pending status in place (a workover completion coming due).

        :param row: The connection row to write.
        :param status: The new active/pending status.
        """
        self.schedule_statuses[row] = 1 if status == WellStatus.ACTIVE else 0

    def get_skin(self, *, row: Integer) -> float:
        """
        :param row: The connection row to read.
        :returns: This connection's own skin factor.
        """
        return float(self.skins[row])

    def set_skin(self, *, row: Integer, value: Number) -> None:
        """
        Overwrites a connection's skin factor in place.

        :param row: The connection row to write.
        :param value: The new skin factor.
        """
        self.skins[row] = value

    def get_wellbore_radius(self, *, row: Integer) -> float:
        """
        :param row: The connection row to read.
        :returns: This connection's own wellbore radius.
        """
        return float(self.wellbore_radii[row])

    def set_wellbore_radius(self, *, row: Integer, value: Number) -> None:
        """
        Overwrites a connection's wellbore radius in place.

        :param row: The connection row to write.
        :param value: The new wellbore radius.
        """
        self.wellbore_radii[row] = value

    def get_saturation_region(self, *, row: Integer) -> int | None:
        """
        :param row: The connection row to read.
        :returns: This connection's own SATNUM override, or `None` to use the cell's own region.
        """
        region = int(self.saturation_regions[row])
        return None if region == UNSET_INT else region


class CompiledLimits(typing.NamedTuple):
    """
    Secondary well limits, flattened row-per-limit and CSR-indexed by
    well, aligned 1:1 with `CompiledWellControls`' rows. A well with no
    limits has an empty row range.
    """

    well_offsets: IntArray[OneDimension]
    """Shape `(n_wells + 1,)`."""

    kinds: IntArray[OneDimension]
    """Shape `(n_rows,)`."""

    quantities: IntArray[OneDimension]
    """
    Shape `(n_rows,)`. A `RateQuantityTag` on a `RATE` row, an
    `EconomicQuantityTag` on an `ECONOMIC` row, `UNSET_INT` otherwise.
    """

    min_values: NumberArray[OneDimension]
    """Shape `(n_rows,)`. `NaN` where this limit has no floor."""

    max_values: NumberArray[OneDimension]
    """Shape `(n_rows,)`. `NaN` where this limit has no ceiling."""

    workover_actions: IntArray[OneDimension]
    """Shape `(n_rows,)`. A `WorkoverActionTag` on an `ECONOMIC` row,
    `UNSET_INT` otherwise - what to do once that row is breached."""

    end_run_flags: IntArray[OneDimension]
    """Shape `(n_rows,)`. `1` on an `ECONOMIC` row that should stop the
    whole run once breached, `0` otherwise."""

    def limit_rows(self, *, well_row: Integer) -> range:
        """
        A well's own limit-row range, without the caller needing to know about `well_offsets`.

        :param well_row: The well's row in `CompiledWellSystem.names`.
        :returns: `range(start, end)` over this well's limit rows.
        """
        return range(int(self.well_offsets[well_row]), int(self.well_offsets[well_row + 1]))

    def find_economic_limit_row(self, *, well_row: Integer) -> int | None:
        """
        Finds a well's own `ECONOMIC`-kind limit row, if it has one.

        :param well_row: The well's row.
        :returns: The row, or `None` if this well has no economic limit.
        """
        for row in self.limit_rows(well_row=well_row):
            if self.kinds[row] == LimitKind.ECONOMIC:
                return row
        return None

    def get_kind(self, *, row: Integer) -> LimitKind:
        """
        :param row: The limit row to read.
        :returns: This row's own kind.
        """
        return LimitKind(int(self.kinds[row]))

    def get_min_value(self, *, row: Integer) -> float | None:
        """
        :param row: The limit row to read.
        :returns: This row's own floor, or `None` if it has none.
        """
        return none_if_nan(float(self.min_values[row]))

    def set_min_value(self, *, row: Integer, value: Number) -> None:
        """
        Overwrites a limit row's floor in place.

        :param row: The limit row to write.
        :param value: The new floor.
        """
        self.min_values[row] = value

    def get_max_value(self, *, row: Integer) -> float | None:
        """
        :param row: The limit row to read.
        :returns: This row's own ceiling, or `None` if it has none.
        """
        return none_if_nan(float(self.max_values[row]))

    def set_max_value(self, *, row: Integer, value: Number) -> None:
        """
        Overwrites a limit row's ceiling in place.

        :param row: The limit row to write.
        :param value: The new ceiling.
        """
        self.max_values[row] = value

    def get_workover_action(self, *, row: Integer) -> WorkoverAction | None:
        """
        :param row: The limit row to read.
        :returns: This `ECONOMIC` row's own workover action, or `None` on a non-economic row.
        """
        tag = self.workover_actions[row]
        return None if tag == UNSET_INT else WORKOVER_ACTION_FROM_TAG[tag]

    def set_workover_action(self, *, row: Integer, action: WorkoverAction) -> None:
        """
        Overwrites an `ECONOMIC` row's workover action in place.

        :param row: The limit row to write.
        :param action: The new workover action.
        """
        self.workover_actions[row] = WORKOVER_ACTION_TAG[action]

    def get_end_run(self, *, row: Integer) -> bool:
        """
        :param row: The limit row to read.
        :returns: Whether breaching this row should stop the whole run.
        """
        return bool(self.end_run_flags[row])

    def set_end_run(self, *, row: Integer, end_run: bool) -> None:
        """
        Overwrites an `ECONOMIC` row's end-run flag in place.

        :param row: The limit row to write.
        :param end_run: Whether breaching this row should stop the whole run.
        """
        self.end_run_flags[row] = 1 if end_run else 0


class CompiledWellControls(typing.NamedTuple):
    """
    Every well's current control target, one row per well, positionally
    aligned with `CompiledWellSystem`.
    """

    well_kinds: IntArray[OneDimension]
    """Shape `(n_wells,)`."""

    control_modes: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. A `ProducerControlModeTag` on a `PRODUCER` row,
    an `InjectorControlModeTag` on an `INJECTOR` row. Check `well_kinds`
    first. `UNSET` for a well with no control yet.
    """

    injected_phases: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. A `FluidPhaseTag` on an `INJECTOR` row with a
    control set; `UNSET_INT` otherwise.
    """

    target_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. `NaN` where unset. Whichever rate `control_modes` currently selects."""

    target_bhps: NumberArray[OneDimension]
    """Shape `(n_wells,)`. `NaN` where unset."""

    target_thps: NumberArray[OneDimension]
    """Shape `(n_wells,)`. `NaN` where unset."""

    efficiency_factors: NumberArray[OneDimension]
    """Shape `(n_wells,)`."""

    guide_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. `NaN` where unset."""

    limits: CompiledLimits

    def get_control_mode(self, *, well_row: Integer) -> ProducerControlMode | InjectorControlMode:
        """
        :param well_row: The well's row.
        :returns: This well's own control mode.
        """
        is_injector = self.well_kinds[well_row] == WellKind.INJECTOR
        from_tag = INJECTOR_MODE_FROM_TAG if is_injector else PRODUCER_MODE_FROM_TAG
        return from_tag[self.control_modes[well_row]]

    def set_control_mode(
        self, *, well_row: Integer, mode: ProducerControlMode | InjectorControlMode
    ) -> None:
        """
        Sets a well's control mode in place. `mode` must match the well's own kind.

        :param well_row: The well's row.
        :param mode: The new control mode.
        """
        is_injector = self.well_kinds[well_row] == WellKind.INJECTOR
        tag_map = INJECTOR_MODE_TAG if is_injector else PRODUCER_MODE_TAG
        self.control_modes[well_row] = tag_map[mode]  # type: ignore

    def get_injected_phase(self, *, well_row: Integer) -> FluidPhase | None:
        """
        :param well_row: The well's row.
        :returns: This injector's own injected phase, or `None` on a producer or unset injector.
        """
        tag = self.injected_phases[well_row]
        return None if tag == UNSET_INT else FLUID_PHASE_FROM_TAG[tag]

    def set_injected_phase(self, *, well_row: Integer, phase: FluidPhase) -> None:
        """
        Overwrites an injector's injected phase in place.

        :param well_row: The well's row.
        :param phase: The new injected phase.
        """
        self.injected_phases[well_row] = FLUID_PHASE_TAG[phase]

    def get_target_rate(self, *, well_row: Integer) -> float | None:
        """
        :param well_row: The well's row.
        :returns: This well's own target rate, or `None` if unset.
        """
        return none_if_nan(float(self.target_rates[well_row]))

    def set_target_rate(self, *, well_row: Integer, value: Number) -> None:
        """
        Overwrites a well's target rate in place.

        :param well_row: The well's row.
        :param value: The new target rate.
        """
        self.target_rates[well_row] = value

    def get_target_bhp(self, *, well_row: Integer) -> float | None:
        """
        :param well_row: The well's row.
        :returns: This well's own target BHP, or `None` if unset.
        """
        return none_if_nan(float(self.target_bhps[well_row]))

    def set_target_bhp(self, *, well_row: Integer, value: Number) -> None:
        """
        Overwrites a well's target BHP in place.

        :param well_row: The well's row.
        :param value: The new target BHP.
        """
        self.target_bhps[well_row] = value

    def get_target_thp(self, *, well_row: Integer) -> float | None:
        """
        :param well_row: The well's row.
        :returns: This well's own target THP, or `None` if unset.
        """
        return none_if_nan(float(self.target_thps[well_row]))

    def set_target_thp(self, *, well_row: Integer, value: Number) -> None:
        """
        Overwrites a well's target THP in place.

        :param well_row: The well's row.
        :param value: The new target THP.
        """
        self.target_thps[well_row] = value

    def get_efficiency_factor(self, *, well_row: Integer) -> float:
        """
        :param well_row: The well's row.
        :returns: This well's own efficiency factor.
        """
        return float(self.efficiency_factors[well_row])

    def set_efficiency_factor(self, *, well_row: Integer, value: Number) -> None:
        """
        Overwrites a well's efficiency factor in place.

        :param well_row: The well's row.
        :param value: The new efficiency factor.
        """
        self.efficiency_factors[well_row] = value

    def get_guide_rate(self, *, well_row: Integer) -> float | None:
        """
        :param well_row: The well's row.
        :returns: This well's own guide rate, or `None` if unset.
        """
        return none_if_nan(float(self.guide_rates[well_row]))

    def set_guide_rate(self, *, well_row: Integer, value: Number) -> None:
        """
        Overwrites a well's guide rate in place.

        :param well_row: The well's row.
        :param value: The new guide rate.
        """
        self.guide_rates[well_row] = value


class CompiledGroupControls(typing.NamedTuple):
    """
    Every group's current control target, one row per group, plus each
    group's compiled well membership (direct or via a descendant group).

    A compute once, patch on event quantity like `CompiledPerforations`'
    geometry, since the group hierarchy is deck-defined and doesn't change
    at solve time.
    """

    names: tuple[str, ...]
    """Shape `(n_groups,)`."""

    group_kinds: IntArray[OneDimension]
    """Shape `(n_groups,)`."""

    control_modes: IntArray[OneDimension]
    """
    Shape `(n_groups,)`. A `GroupProducerControlModeTag` on a
    `PRODUCER` row, a `GroupInjectorControlModeTag` on an `INJECTOR` row.
    Check `group_kinds` first.
    """

    injected_phases: IntArray[OneDimension]
    """Shape `(n_groups,)`. A `FluidPhaseTag` on an `INJECTOR` row; `UNSET_INT` otherwise."""

    target_rates: NumberArray[OneDimension]
    """Shape `(n_groups,)`. `NaN` where unset."""

    member_offsets: IntArray[OneDimension]
    """
    Shape `(n_groups + 1,)`. Group `g`'s member well positions are
    `member_well_indices[member_offsets[g]:member_offsets[g + 1]]`. 
    
    A member is any well whose own group is this group or a descendant of
    it (`WellGroups.descendants`), regardless of that well's current
    control mode. Membership is static; mode eligibility (`GRUP` or not)
    is dynamic and checked at allocation time instead.
    """

    member_well_indices: IntArray[OneDimension]
    """
    Shape `(n_members,)`. Positions into `CompiledWellSystem.names`/
    `.controls` (not well names). Direct array indices so there's no further 
    lookup needed at allocation time.
    """


class CompiledWellSystem(typing.NamedTuple):
    """
    Top-level compiled wells bundle for the solver hot path.

    Pass `.perforations`/`.controls`/`.group_controls` (or their individual
    arrays) into a jitted kernel and never this whole tuple, since `names`
    and `unit_system` aren't valid jittable argument types.
    """

    names: tuple[str, ...]
    """Shape `(n_wells,)`. Every other array here is positional against this order."""

    well_kinds: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. Duplicated from `controls.well_kinds`, kept in
    lockstep by construction, for a kernel that needs well identity
    without threading `controls` through too.
    """

    schedule_statuses: IntArray[OneDimension]
    """Shape `(n_wells,)`."""

    reference_depths: NumberArray[OneDimension]
    """Shape `(n_wells,)`."""

    perforations: CompiledPerforations
    """Compiled perforation data for the solver hot path."""

    controls: CompiledWellControls
    """Compiled well control data aligned with `names` and `well_kinds`."""

    group_controls: CompiledGroupControls | None
    """Compiled group-control data, or `None` when no group controls exist."""

    unit_system: UnitSystem
    """Unit system used to interpret the compiled well data."""

    def well_row(self, *, name: str) -> int:
        """
        A well's row, by name.

        :param name: The well to find.
        :returns: The well's row.
        :raises ValidationError: If no well in this system has this name.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise ValidationError(f"No well named {name!r} in this compiled system.") from None

    def has_well(self, *, name: str) -> bool:
        """
        :param name: The well to check for.
        :returns: Whether a well with this name exists in this system.
        """
        return name in self.names

    def get_well_type(self, *, well_row: Integer) -> WellType:
        """
        :param well_row: The well's row.
        :returns: This well's own type.
        """
        return (
            WellType.PRODUCER
            if self.well_kinds[well_row] == WellKind.PRODUCER
            else WellType.INJECTOR
        )

    def get_schedule_status(self, *, well_row: Integer) -> WellStatus:
        """
        :param well_row: The well's row.
        :returns: This well's own active/pending status.
        """
        return WellStatus.ACTIVE if self.schedule_statuses[well_row] else WellStatus.PENDING

    def set_schedule_status(self, *, well_row: Integer, status: WellStatus) -> None:
        """
        Sets a well's active/pending status in place.

        :param well_row: The well's row.
        :param status: The new active/pending status.
        """
        self.schedule_statuses[well_row] = get_well_status_tag(status=status)

    def get_reference_depth(self, *, well_row: Integer) -> float:
        """
        :param well_row: The well's row.
        :returns: This well's own reference depth.
        """
        return float(self.reference_depths[well_row])


def get_well_status_tag(status: WellStatus) -> Integer:
    """
    Converts a `WellStatus` to its compiled tag value.

    :param status: `WellStatus` to convert.
    :returns: `1` for `ACTIVE`, `0` for `PENDING`.
    """
    return 1 if status is WellStatus.ACTIVE else 0


def get_completion_status_tag(status: CompletionStatus) -> Integer:
    """
    Converts a `CompletionStatus` to its compiled tag value.

    :param status: `CompletionStatus` to convert.
    :returns: `1` for `OPEN`, `0` for `SHUT`.
    """
    return 1 if status is CompletionStatus.OPEN else 0


def resolve_perforations_geometry(
    grid: Grid,
    well: Well,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    **resolve_kwargs: typing.Any,
) -> tuple[tuple[PerforationIndex, ...], dict[Integer, AnyPerforation], dict[Integer, int]]:
    """
    Resolves connection geometry and connection factor for every
    perforation on a well, open or shut.

    `resolve_perforations_indices`/`resolve_md_perforations_indices` only
    resolve `well.open_perforations`, so this builds a shadow copy of
    `well` with every perforation forced `CompletionStatus.OPEN` for
    resolution purposes only, then reads each connection's real status
    back off the matching original perforation.

    :param grid: Grid to resolve against.
    :param well: The well to resolve. Not modified.
    :param permeabilities: Forwarded to `build_wells_indices`.
    :param resolve_kwargs: Forwarded to `build_wells_indices`.
    :returns: Every resolved `PerforationIndex` for this well (against the
        shadow perforations), a mapping from `id(shadow_perforation)` to
        the matching original perforation, and a mapping from
        `id(shadow_perforation)` to that perforation's ordinal position in
        `well.perforations`.
    """
    shadow_perforations = tuple(
        attrs.evolve(perforation, status=CompletionStatus.OPEN)
        for perforation in well.perforations
    )
    original_by_id: dict[Integer, AnyPerforation] = {
        id(shadow): original
        for shadow, original in zip(shadow_perforations, well.perforations, strict=False)
    }
    ordinal_by_id: dict[Integer, int] = {
        id(shadow): ordinal for ordinal, shadow in enumerate(shadow_perforations)
    }
    shadow_well = attrs.evolve(well, perforations=shadow_perforations)
    shadow_wells = Wells(wells={well.name: shadow_well})

    result = build_wells_indices(
        grid=grid,
        wells=shadow_wells,
        permeabilities=permeabilities,
        **resolve_kwargs,
    )
    well_index = result[well.name]
    return well_index.perforations, original_by_id, ordinal_by_id


def compile_perforations(
    names: typing.Sequence[str],
    wells: Wells,
    grid: Grid,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    dtype: npt.DTypeLike = None,
    **resolve_kwargs: typing.Any,
) -> CompiledPerforations:
    """
    Builds `CompiledPerforations` for a set of wells, in order.

    :param names: Well names, in the row order `CompiledWellSystem` will use.
    :param wells: Source `Wells`.
    :param grid: Forwarded to `resolve_perforations_geometry`.
    :param permeabilities: Forwarded to `resolve_perforations_geometry`.
    :param resolve_kwargs: Forwarded to `resolve_perforations_geometry`.
    :returns: `CompiledPerforations`, one row per (well, perforation, cell) triple.
    """
    well_offsets = [0]
    cell_indices: list[Integer] = []
    perforation_indices: list[Integer] = []
    well_indices: list[Number] = []
    wellbore_radii: list[Number] = []
    skins: list[Number] = []
    partial_penetration_fractions: list[Number] = []
    representative_depths: list[Number] = []
    inclinations_from_vertical: list[Number] = []
    completion_statuses: list[Integer] = []
    schedule_statuses: list[Integer] = []
    saturation_regions: list[Integer] = []

    for name in names:
        well = wells[name]
        perforation_indices_geometry, original_by_id, ordinal_by_id = (
            resolve_perforations_geometry(
                grid=grid,
                well=well,
                permeabilities=permeabilities,
                **resolve_kwargs,
            )
        )
        for perforation_index in perforation_indices_geometry:
            shadow_id = id(perforation_index.perforation)
            original = original_by_id[shadow_id]
            cell_indices.append(perforation_index.cell_index)
            perforation_indices.append(ordinal_by_id[shadow_id])
            well_indices.append(
                perforation_index.well_index
                if perforation_index.well_index is not None
                else np.nan
            )
            wellbore_radii.append(original.wellbore_radius)
            skins.append(original.skin)
            partial_penetration_fractions.append(perforation_index.partial_penetration_fraction)
            representative_depths.append(perforation_index.representative_depth)
            inclinations_from_vertical.append(perforation_index.inclination_from_vertical)
            completion_statuses.append(get_completion_status_tag(status=original.status))
            schedule_statuses.append(get_well_status_tag(status=original.schedule_status))
            saturation_regions.append(
                original.saturation_region if original.saturation_region is not None else UNSET_INT
            )
        well_offsets.append(len(cell_indices))

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    return CompiledPerforations(
        well_offsets=typing.cast(IntArray[OneDimension], np.asarray(well_offsets, dtype=np.int64)),
        cell_indices=typing.cast(IntArray[OneDimension], np.asarray(cell_indices, dtype=np.int64)),
        perforation_indices=typing.cast(
            IntArray[OneDimension], np.asarray(perforation_indices, dtype=np.int64)
        ),
        well_indices=typing.cast(NumberArray[OneDimension], np.asarray(well_indices, dtype=dtype)),
        wellbore_radii=typing.cast(
            NumberArray[OneDimension], np.asarray(wellbore_radii, dtype=dtype)
        ),
        skins=typing.cast(NumberArray[OneDimension], np.asarray(skins, dtype=dtype)),
        partial_penetration_fractions=typing.cast(
            NumberArray[OneDimension],
            np.asarray(partial_penetration_fractions, dtype=dtype),
        ),
        representative_depths=typing.cast(
            NumberArray[OneDimension], np.asarray(representative_depths, dtype=dtype)
        ),
        inclinations_from_vertical=typing.cast(
            NumberArray[OneDimension],
            np.asarray(inclinations_from_vertical, dtype=dtype),
        ),
        completion_statuses=typing.cast(
            IntArray[OneDimension],
            np.asarray(completion_statuses, dtype=np.int32),
        ),
        schedule_statuses=typing.cast(
            IntArray[OneDimension], np.asarray(schedule_statuses, dtype=np.int32)
        ),
        saturation_regions=typing.cast(
            IntArray[OneDimension], np.asarray(saturation_regions, dtype=np.int32)
        ),
    )


def _compile_limits(
    limits: typing.Sequence[Limit],
) -> tuple[list[Integer], list[Integer], list[Number], list[Number], list[Integer], list[Integer]]:
    """
    Flattens one well's or group's limits into parallel arrays.

    :param limits: Limits to flatten.
    :returns: `(kinds, quantities, min_values, max_values,
        workover_actions, end_run_flags)`, all the same length as `limits`.
    :raises ValidationError: If `limits` contains an unrecognized `Limit` subtype.
    """
    kinds: list[Integer] = []
    quantities: list[Integer] = []
    min_values: list[Number] = []
    max_values: list[Number] = []
    workover_actions: list[Integer] = []
    end_run_flags: list[Integer] = []

    for limit in limits:
        if isinstance(limit, BHPLimit):
            kinds.append(LimitKind.BHP)
            quantities.append(UNSET_INT)
            min_values.append(limit.min_value if limit.min_value is not None else np.nan)
            max_values.append(limit.max_value if limit.max_value is not None else np.nan)
            workover_actions.append(UNSET_INT)
            end_run_flags.append(0)
        elif isinstance(limit, THPLimit):
            kinds.append(LimitKind.THP)
            quantities.append(UNSET_INT)
            min_values.append(limit.min_value if limit.min_value is not None else np.nan)
            max_values.append(limit.max_value if limit.max_value is not None else np.nan)
            workover_actions.append(UNSET_INT)
            end_run_flags.append(0)
        elif isinstance(limit, RateLimit):
            kinds.append(LimitKind.RATE)
            quantities.append(RATE_QUANTITY_TAG[limit.quantity])
            min_values.append(np.nan)
            max_values.append(limit.max_value)
            workover_actions.append(UNSET_INT)
            end_run_flags.append(0)
        elif isinstance(limit, EconomicLimit):
            kinds.append(LimitKind.ECONOMIC)
            quantities.append(ECONOMIC_QUANTITY_TAG[limit.quantity])
            min_values.append(limit.min_value if limit.min_value is not None else np.nan)
            max_values.append(limit.max_value if limit.max_value is not None else np.nan)
            workover_actions.append(WORKOVER_ACTION_TAG[limit.workover_action])
            end_run_flags.append(1 if limit.end_run else 0)
        else:
            raise ValidationError(f"Unknown Limit type: {type(limit)!r}.")

    return kinds, quantities, min_values, max_values, workover_actions, end_run_flags


def compile_well_controls(
    names: typing.Sequence[str],
    controls: WellControls,
    wells: Wells,
    dtype: npt.DTypeLike = None,
) -> CompiledWellControls:
    """
    Builds `CompiledWellControls` for a set of wells, in order.

    A well with no entry in `controls` gets an `UNSET`-mode row with every
    target as `NaN` and `efficiency_factor=1.0`, rather than an erroring since an
    uncontrolled well is a normal roster state, not a data problem.

    Shared fields (`target_rate`, `target_bhp`, `target_thp`, `efficiency_factor`,
    `guide_rate`, `limits`) are read via `getattr` with a default rather than assumed
    present, since `WellControl` itself is an abstract base declaring none of them,
    as only its concrete subclasses do.

    :param names: Well names, in row order.
    :param controls: Source `WellControls`.
    :param wells: Source `Wells`. Only `well_type` is read, to set
        `well_kinds` even for a well with no control yet.
    :returns: `CompiledWellControls`.
    :raises ValidationError: If a control is neither a `ProducerControl` nor `InjectorControl`.
    """
    well_kinds: list[Integer] = []
    control_modes: list[Integer] = []
    injected_phases: list[Integer] = []
    target_rates: list[Number] = []
    target_bhps: list[Number] = []
    target_thps: list[Number] = []
    efficiency_factors: list[Number] = []
    guide_rates: list[Number] = []

    limits_well_offsets = [0]
    limits_kinds: list[Integer] = []
    limits_quantities: list[Integer] = []
    limits_min_values: list[Number] = []
    limits_max_values: list[Number] = []
    limits_workover_actions: list[Integer] = []
    limits_end_run_flags: list[Integer] = []

    for name in names:
        well = wells[name]
        control = controls.get(name)
        kind = WellKind.INJECTOR if well.well_type is WellType.INJECTOR else WellKind.PRODUCER
        well_kinds.append(kind)

        if control is None:
            control_modes.append(
                ProducerControlModeTag.UNSET
                if kind is WellKind.PRODUCER
                else InjectorControlModeTag.UNSET
            )
            injected_phases.append(UNSET_INT)
            target_rates.append(np.nan)
            target_bhps.append(np.nan)
            target_thps.append(np.nan)
            efficiency_factors.append(1.0)
            guide_rates.append(np.nan)
            limits_well_offsets.append(len(limits_kinds))
            continue

        target_bhp = getattr(control, "target_bhp", None)
        target_thp = getattr(control, "target_thp", None)
        target_rate = getattr(control, "target_rate", None)
        efficiency_factor = getattr(control, "efficiency_factor", 1.0)
        guide_rate = getattr(control, "guide_rate", None)
        control_limits = getattr(control, "limits", ())

        efficiency_factors.append(efficiency_factor)
        guide_rates.append(guide_rate if guide_rate is not None else np.nan)
        target_bhps.append(target_bhp if target_bhp is not None else np.nan)
        target_thps.append(target_thp if target_thp is not None else np.nan)
        target_rates.append(target_rate if target_rate is not None else np.nan)

        if isinstance(control, InjectorControl):
            control_modes.append(INJECTOR_MODE_TAG[control.mode])
            injected_phases.append(FLUID_PHASE_TAG[control.injected_phase])
        elif isinstance(control, ProducerControl):
            control_modes.append(PRODUCER_MODE_TAG[control.mode])
            injected_phases.append(UNSET_INT)
        else:
            raise ValidationError(f"Unknown WellControl type: {type(control)!r}.")

        kinds, quantities, min_values, max_values, workover_actions, end_run_flags = (
            _compile_limits(limits=control_limits)
        )
        limits_kinds.extend(kinds)
        limits_quantities.extend(quantities)
        limits_min_values.extend(min_values)
        limits_max_values.extend(max_values)
        limits_workover_actions.extend(workover_actions)
        limits_end_run_flags.extend(end_run_flags)
        limits_well_offsets.append(len(limits_kinds))

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    limits = CompiledLimits(
        well_offsets=typing.cast(
            IntArray[OneDimension], np.asarray(limits_well_offsets, dtype=np.int64)
        ),
        kinds=typing.cast(IntArray[OneDimension], np.asarray(limits_kinds, dtype=np.int32)),
        quantities=typing.cast(
            IntArray[OneDimension], np.asarray(limits_quantities, dtype=np.int32)
        ),
        min_values=typing.cast(
            NumberArray[OneDimension], np.asarray(limits_min_values, dtype=dtype)
        ),
        max_values=typing.cast(
            NumberArray[OneDimension], np.asarray(limits_max_values, dtype=dtype)
        ),
        workover_actions=typing.cast(
            IntArray[OneDimension], np.asarray(limits_workover_actions, dtype=np.int32)
        ),
        end_run_flags=typing.cast(
            IntArray[OneDimension], np.asarray(limits_end_run_flags, dtype=np.int32)
        ),
    )
    return CompiledWellControls(
        well_kinds=typing.cast(IntArray[OneDimension], np.asarray(well_kinds, dtype=np.int32)),
        control_modes=typing.cast(
            IntArray[OneDimension], np.asarray(control_modes, dtype=np.int32)
        ),
        injected_phases=typing.cast(
            IntArray[OneDimension], np.asarray(injected_phases, dtype=np.int32)
        ),
        target_rates=typing.cast(NumberArray[OneDimension], np.asarray(target_rates, dtype=dtype)),
        target_bhps=typing.cast(NumberArray[OneDimension], np.asarray(target_bhps, dtype=dtype)),
        target_thps=typing.cast(NumberArray[OneDimension], np.asarray(target_thps, dtype=dtype)),
        efficiency_factors=typing.cast(
            NumberArray[OneDimension], np.asarray(efficiency_factors, dtype=dtype)
        ),
        guide_rates=typing.cast(NumberArray[OneDimension], np.asarray(guide_rates, dtype=dtype)),
        limits=limits,
    )


def compile_group_controls(
    group_controls: GroupControls | None,
    wells: Wells,
    names: typing.Sequence[str],
    groups: WellGroups | None = None,
    dtype: npt.DTypeLike = None,
) -> CompiledGroupControls | None:
    """
    Builds `CompiledGroupControls` from a `GroupControls`, including each
    control's compiled well membership.

    Only compiles groups that have an explicit control - it does not
    cross-reference `WellGroups`' hierarchy for which groups *exist*, so a
    group with no control of its own is simply absent from the result, not
    included as an `UNSET`-mode row the way an uncontrolled well is in
    `compile_well_controls`. Membership *is* resolved through the
    hierarchy (a well belongs to a controlled group if its own group is
    that group or any descendant of it), independent of whether those
    descendant groups have controls of their own.

    :param group_controls: Source `GroupControls`, or `None`.
    :param wells: Every well in the system - `Well.group` supplies each
        well's own (direct) group name.
    :param names: Well names in `CompiledWellSystem.names` order -
        `member_well_indices` positions are indices into this sequence.
    :param groups: The group hierarchy, for descendant resolution. `None`
        restricts membership to a group's *direct* members only (no
        `WellGroups` to look up descendants through).
    :returns: `CompiledGroupControls`, one row per group with a control,
        in sorted name order; `None` if `group_controls` is `None`.
    :raises ValidationError: If a group's control mode isn't a recognized
        `GroupProducerControlMode`/`GroupInjectorControlMode`.
    """
    if group_controls is None:
        return None

    control_names = sorted(group_controls.controls.keys())
    group_kinds: list[Integer] = []
    control_modes: list[Integer] = []
    injected_phases: list[Integer] = []
    target_rates: list[Number] = []

    name_to_index = {name: i for i, name in enumerate(names)}
    member_offsets: list[Integer] = [0]
    member_well_indices: list[Integer] = []

    for name in control_names:
        control = group_controls[name]
        if isinstance(control.mode, GroupInjectorControlMode):
            group_kinds.append(GroupKind.INJECTOR)
            control_modes.append(GROUP_INJECTOR_MODE_TAG[control.mode])
            injected_phases.append(
                FLUID_PHASE_TAG[control.injected_phase]
                if control.injected_phase is not None
                else UNSET_INT
            )
        elif isinstance(control.mode, GroupProducerControlMode):
            group_kinds.append(GroupKind.PRODUCER)
            control_modes.append(GROUP_PRODUCER_MODE_TAG[control.mode])
            injected_phases.append(UNSET_INT)
        else:
            raise ValidationError(f"Unknown GroupControl mode type: {type(control.mode)!r}.")
        target_rates.append(control.target_rate if control.target_rate is not None else np.nan)

        member_group_names = {name, *groups.descendants(name)} if groups is not None else {name}
        for well_name in names:
            if wells[well_name].group in member_group_names:
                member_well_indices.append(name_to_index[well_name])
        member_offsets.append(len(member_well_indices))

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    return CompiledGroupControls(
        names=tuple(control_names),
        group_kinds=typing.cast(IntArray[OneDimension], np.asarray(group_kinds, dtype=np.int32)),
        control_modes=typing.cast(
            IntArray[OneDimension], np.asarray(control_modes, dtype=np.int32)
        ),
        injected_phases=typing.cast(
            IntArray[OneDimension], np.asarray(injected_phases, dtype=np.int32)
        ),
        target_rates=typing.cast(NumberArray[OneDimension], np.asarray(target_rates, dtype=dtype)),
        member_offsets=typing.cast(
            IntArray[OneDimension], np.asarray(member_offsets, dtype=np.int32)
        ),
        member_well_indices=typing.cast(
            IntArray[OneDimension], np.asarray(member_well_indices, dtype=np.int32)
        ),
    )


def compile_well_system(
    wells: Wells,
    controls: WellControls,
    grid: Grid,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    group_controls: GroupControls | None = None,
    groups: WellGroups | None = None,
    dtype: npt.DTypeLike = None,
    **resolve_kwargs: typing.Any,
) -> CompiledWellSystem:
    """
    Compiles a rich `Wells`/`WellControls` pair into a `CompiledWellSystem`.

    Every well `wells` contains is included, regardless of `schedule_status`
    as this is the load-once roster: a `WellStatus.PENDING` well is compiled
    the same as an `ACTIVE` one, just tagged, so a later schedule event can
    activate it in place without recompiling.

    :param wells: Every well that will ever exist across the run.
    :param controls: Current controls. A well absent from `controls`
        compiles to an `UNSET`-mode row (see `compile_well_controls`).
    :param grid: Grid to resolve perforation geometry against.
    :param permeabilities: Per-axis permeability arrays, forwarded to
        `wells.indices.wells.build_wells_indices`.
    :param group_controls: Current group controls, if any.
    :param groups: The group hierarchy, if any - `WellSystem.groups`.
        Forwarded to `compile_group_controls` for descendant-aware
        membership; a group control without a hierarchy still compiles,
        with membership restricted to that group's direct members.
    :param resolve_kwargs: Forwarded to `build_wells_indices`.
    :returns: `CompiledWellSystem`, well rows in `wells.names` order.
    :raises ValidationError: If `controls.unit_system` doesn't match `wells.unit_system`.
    """
    if controls.unit_system != wells.unit_system:
        raise ValidationError(
            f"`controls.unit_system` ({controls.unit_system.value}) != "
            f"`wells.unit_system` ({wells.unit_system.value}). Convert both "
            "to the same unit system before compiling."
        )

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    names = wells.names
    perforations = compile_perforations(
        names=names,
        wells=wells,
        grid=grid,
        permeabilities=permeabilities,
        dtype=dtype,
        **resolve_kwargs,
    )
    well_controls = compile_well_controls(
        names=names,
        controls=controls,
        wells=wells,
        dtype=dtype,
    )

    schedule_statuses = typing.cast(
        IntArray[OneDimension],
        np.asarray(
            [get_well_status_tag(status=wells[name].schedule_status) for name in names],
            dtype=np.int32,
        ),
    )
    reference_depths = typing.cast(
        NumberArray[OneDimension],
        np.asarray([wells[name].reference_depth for name in names], dtype=dtype),
    )
    return CompiledWellSystem(
        names=tuple(names),
        well_kinds=well_controls.well_kinds,
        schedule_statuses=schedule_statuses,
        reference_depths=reference_depths,
        perforations=perforations,
        controls=well_controls,
        group_controls=compile_group_controls(
            group_controls=group_controls,
            wells=wells,
            names=names,
            groups=groups,
            dtype=dtype,
        ),
        unit_system=wells.unit_system,
    )
