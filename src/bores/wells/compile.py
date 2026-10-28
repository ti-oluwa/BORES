"""Compiled (Structure-of-Arrays) well data for the reservoir solver hot path."""

import enum
import functools
import typing

import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.types import (
    Boolean,
    FluidPhase,
    GridIntersectionMethod,
    IntArray,
    Integer,
    IntOrArray,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    Orientation,
    UnitSystem,
)
from bores.utils import none_if_nan
from bores.wells.base import (
    CompletionStatus,
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
from bores.wells.indices.wells import build_well_index

__all__ = [
    "CompiledGroupControls",
    "CompiledGroupLimits",
    "CompiledLimits",
    "CompiledPerforations",
    "CompiledWellControls",
    "CompiledWellSystem",
    "GroupKind",
    "LimitColumns",
    "LimitKind",
    "WellKind",
    "compile_group_controls",
    "compile_perforations",
    "compile_well_controls",
    "compile_well_system",
]


UNSET_INT = -1
"""Sentinel for an int tag column entry that doesn't apply to that row."""


@functools.lru_cache(maxsize=32)
def name_to_row_map(names: tuple[str, ...]) -> dict[str, int]:
    """
    Builds (and caches) a name-to-row dict for a `names` tuple.

    `CompiledWellSystem`/`CompiledGroupControls` are frozen `NamedTuple`s,
    so a given `names` tuple is a stable, hashable identity for the
    lifetime of the system it belongs to - the cache key. `well_row`/
    `CompiledGroupControls.group_row` call this instead of `names.index`,
    trading one O(n) dict build (once, per distinct `names` tuple) for
    every lookup after that being O(1) instead of O(n).

    :param names: A `CompiledWellSystem.names` or
        `CompiledGroupControls.names` tuple.
    :returns: `{name: row}` for every entry in `names`.
    """
    return {name: row for row, name in enumerate(names)}


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
    NONE = 4
    """`GECON`-only. Tracked but not enforced."""
    RATE = 5
    """`GECON`-only. Cut the group's rate rather than shutting a well."""


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
    WorkoverAction.NONE: WorkoverActionTag.NONE,
    WorkoverAction.RATE: WorkoverActionTag.RATE,
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


class CompiledPerforations(typing.NamedTuple):
    """
    Every well's perforations, flattened row-per-connection and
    CSR-indexed by well. Includes every perforation regardless of status.

    A shut or not yet scheduled connection is a tagged row here, and is not
    absent, so a later status change never needs to re-resolve geometry.
    """

    well_offsets: IntArray[OneDimension]
    """Shape `(n_wells + 1,)`. Well `w`'s rows are `well_offsets[w]:well_offsets[w + 1]`."""

    cell_indices: IntArray[OneDimension]
    """Shape `(n_rows,)`. The grid cell each connection resolves to."""

    perforation_indices: IntArray[OneDimension]
    """
    Shape `(n_rows,)`. Position of the rich `Perforation`
    this row came from, within that well's own `Well.perforations`. One
    rich perforation can resolve to several rows (a trajectory crossing
    several grid cells), so this is not the same as the row's own
    position. It lets a row be traced back to its source without relying
    on row position, which decompiling needs since `CompiledPerforations`
    keeps no other reference to the rich perforations it was built from.
    """

    well_indices: NumberArray[OneDimension]
    """Shape `(n_rows,)`. Each connection's connection factor."""

    connection_conductivities: NumberArray[OneDimension]
    """
    Shape `(n_rows,)`. Peaceman numerator (`k*h*N/G`), isolated from
    skin/geometry. `NaN` on an overridden connection
    (`Perforation.connection_factor_override`), where there's no such
    decomposition. Lets a rate-dependent skin addition (non-Darcy `D*q`)
    be applied to `well_indices` without rederiving the well-index formula.
    """

    wellbore_radii: NumberArray[OneDimension]
    """Shape `(n_rows,)`."""

    skins: NumberArray[OneDimension]
    """Shape `(n_rows,)`."""

    partial_penetration_fractions: NumberArray[OneDimension]
    """Shape `(n_rows,)`."""

    representative_depths: NumberArray[OneDimension]
    """Shape `(n_rows,)`. For the hydrostatic term in wellbore pressure-drop."""

    representative_measured_depths: NumberArray[OneDimension]
    """Shape `(n_rows,)`. Along the wellbore, for the friction term in wellbore pressure-drop."""

    inclinations_from_vertical: NumberArray[OneDimension]
    """Shape `(n_rows,)`, radians. For `wells.hydraulics`."""

    completion_statuses: IntArray[OneDimension]
    """Shape `(n_rows,)`. `1` for `CompletionStatus.OPEN`, `0` for `SHUT`."""

    schedule_statuses: IntArray[OneDimension]
    """
    Shape `(n_rows,)`. `1` for `WellStatus.ACTIVE`, `0` for `PENDING`.
    A row with either this or `completion_statuses` at `0` is skipped by
    every solver kernel, as is every row of a well that is pending or shut
    (`CompiledWellSystem.schedule_statuses` and `CompiledWellSystem.open_statuses`).
    """

    saturation_regions: IntArray[OneDimension]
    """Shape `(n_rows,)`. Per-connection SATNUM override; `-1` means use the cell's own region."""

    def connection_rows(self, *, well_row: Integer) -> range:
        """
        A well's own row range, without the caller needing to know about `well_offsets`.

        :param well_row: The well's row in `CompiledWellSystem.names`.
        :returns: `range(start, end)` over this well's rows.
        """
        return range(self.well_offsets[well_row], self.well_offsets[well_row + 1])

    @typing.overload
    def get_cell_index(self, *, row: Integer) -> Integer: ...
    @typing.overload
    def get_cell_index(self, *, row: IntArray[OneDimension]) -> IntArray[OneDimension]: ...
    def get_cell_index(self, *, row: IntOrArray[OneDimension]) -> IntOrArray[OneDimension]:
        """
        :param row: One connection row, or an array of them.
        :returns: The grid cell each connection resolves to. Matches `row`'s own shape.
        """
        return self.cell_indices[row]  # type: ignore[index]

    @typing.overload
    def get_well_index(self, *, row: Integer) -> Number: ...
    @typing.overload
    def get_well_index(self, *, row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_well_index(self, *, row: IntOrArray[OneDimension]) -> NumberOrArray[OneDimension]:
        """
        :param row: One connection row, or an array of them.
        :returns: Each connection's own connection factor (well index). Matches `row`'s own shape.
        """
        return self.well_indices[row]  # type: ignore[index]

    @typing.overload
    def get_connection_conductivity(self, *, row: Integer) -> Number | None: ...
    @typing.overload
    def get_connection_conductivity(
        self, *, row: IntArray[OneDimension]
    ) -> NumberArray[OneDimension]: ...
    def get_connection_conductivity(
        self, *, row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param row: One connection row, or an array of them.
        :returns: Each connection's own Peaceman numerator. For a single
            `row`, a value or `None` if unset (an overridden connection).
            For an array, the raw array (`NaN` means unset).
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            return none_if_nan(self.connection_conductivities[row])
        return self.connection_conductivities[row]  # type: ignore[index]

    def set_well_index(
        self, *, row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more connections' connection factor in place.

        :param row: One connection row, or an array of them.
        :param value: The new connection factor. A single value applies
            to every row in `row`; an array sets each row to its own value.
        """
        self.well_indices[row] = value

    def multiply_well_index(
        self, *, row: IntOrArray[OneDimension], factor: NumberOrArray[OneDimension]
    ) -> None:
        """
        Multiplies one or more connections' existing connection factor in
        place (a `WPIMULT` reissue).

        :param row: One connection row, or an array of them.
        :param factor: The multiplier to apply. A single value applies to
            every row in `row`; an array multiplies each row by its own factor.
        """
        self.well_indices[row] *= factor

    @typing.overload
    def get_completion_status(self, *, row: Integer) -> CompletionStatus: ...
    @typing.overload
    def get_completion_status(self, *, row: IntArray[OneDimension]) -> list[CompletionStatus]: ...
    def get_completion_status(
        self, *, row: IntOrArray[OneDimension]
    ) -> CompletionStatus | list[CompletionStatus]:
        """
        :param row: One connection row, or an array of them.
        :returns: Each connection's own open/shut status. A single
            `CompletionStatus` for a single `row`, a list for an array.
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            return (
                CompletionStatus.OPEN if self.completion_statuses[row] else CompletionStatus.SHUT
            )
        return [
            CompletionStatus.OPEN if tag else CompletionStatus.SHUT
            for tag in self.completion_statuses[row]  # type: ignore[index]
        ]

    def set_completion_status(
        self,
        *,
        row: IntOrArray[OneDimension],
        status: CompletionStatus | typing.Sequence[CompletionStatus],
    ) -> None:
        """
        Sets one or more connections' open/shut status in place (a `WELOPEN` event).

        :param row: One connection row, or an array of them.
        :param status: The new open/shut status. A single `CompletionStatus`
            applies to every row in `row`; a sequence sets each row to its own status.
        """
        if isinstance(status, CompletionStatus):
            self.completion_statuses[row] = 1 if status == CompletionStatus.OPEN else 0
        else:
            self.completion_statuses[row] = [
                1 if one_status == CompletionStatus.OPEN else 0 for one_status in status
            ]

    @typing.overload
    def get_schedule_status(self, *, row: Integer) -> WellStatus: ...
    @typing.overload
    def get_schedule_status(self, *, row: IntArray[OneDimension]) -> list[WellStatus]: ...
    def get_schedule_status(
        self, *, row: IntOrArray[OneDimension]
    ) -> WellStatus | list[WellStatus]:
        """
        :param row: One connection row, or an array of them.
        :returns: Each connection's own active/pending status. A single
            `WellStatus` for a single `row`, a list for an array.
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            return WellStatus.ACTIVE if self.schedule_statuses[row] else WellStatus.PENDING
        return [
            WellStatus.ACTIVE if tag else WellStatus.PENDING
            for tag in self.schedule_statuses[row]  # type: ignore[index]
        ]

    def set_schedule_status(
        self,
        *,
        row: IntOrArray[OneDimension],
        status: WellStatus | typing.Sequence[WellStatus],
    ) -> None:
        """
        Sets one or more connections' active/pending status in place (a
        workover completion coming due).

        :param row: One connection row, or an array of them.
        :param status: The new active/pending status. A single `WellStatus`
            applies to every row in `row`; a sequence sets each row to its own status.
        """
        if isinstance(status, WellStatus):
            self.schedule_statuses[row] = 1 if status == WellStatus.ACTIVE else 0
        else:
            self.schedule_statuses[row] = [
                1 if one_status == WellStatus.ACTIVE else 0 for one_status in status
            ]

    @typing.overload
    def get_skin(self, *, row: Integer) -> Number: ...
    @typing.overload
    def get_skin(self, *, row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_skin(self, *, row: IntOrArray[OneDimension]) -> NumberOrArray[OneDimension]:
        """
        :param row: One connection row, or an array of them.
        :returns: Each connection's own skin factor. Matches `row`'s own shape.
        """
        return self.skins[row]  # type: ignore[index]

    def set_skin(
        self, *, row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more connections' skin factor in place.

        :param row: One connection row, or an array of them.
        :param value: The new skin factor. A single value applies to
            every row in `row`; an array sets each row to its own value.
        """
        self.skins[row] = value

    @typing.overload
    def get_wellbore_radius(self, *, row: Integer) -> Number: ...
    @typing.overload
    def get_wellbore_radius(self, *, row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_wellbore_radius(self, *, row: IntOrArray[OneDimension]) -> NumberOrArray[OneDimension]:
        """
        :param row: One connection row, or an array of them.
        :returns: Each connection's own wellbore radius. Matches `row`'s own shape.
        """
        return self.wellbore_radii[row]  # type: ignore[index]

    def set_wellbore_radius(
        self, *, row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more connections' wellbore radius in place.

        :param row: One connection row, or an array of them.
        :param value: The new wellbore radius. A single value applies to
            every row in `row`; an array sets each row to its own value.
        """
        self.wellbore_radii[row] = value

    @typing.overload
    def get_saturation_region(self, *, row: Integer) -> int | None: ...
    @typing.overload
    def get_saturation_region(self, *, row: IntArray[OneDimension]) -> list[int | None]: ...
    def get_saturation_region(
        self, *, row: IntOrArray[OneDimension]
    ) -> int | list[int | None] | None:
        """
        :param row: One connection row, or an array of them.
        :returns: Each connection's own SATNUM override, or `None` where
            it uses the cell's own region. A single value for a single
            `row`, a list for an array.
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            region = self.saturation_regions[row]
            return None if region == UNSET_INT else region
        return [
            None if region == UNSET_INT else region
            for region in self.saturation_regions[row]  # type: ignore[index]
        ]


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
    `UNSET_INT` otherwise. What to do once that row is breached."""

    end_run_flags: IntArray[OneDimension]
    """Shape `(n_rows,)`. `1` on an `ECONOMIC` row that should stop the
    whole run once breached, `0` otherwise."""

    def limit_rows(self, *, well_row: Integer) -> range:
        """
        A well's own limit-row range, without the caller needing to know about `well_offsets`.

        :param well_row: The well's row in `CompiledWellSystem.names`.
        :returns: `range(start, end)` over this well's limit rows.
        """
        return range(self.well_offsets[well_row], self.well_offsets[well_row + 1])

    def find_limit_row(
        self,
        *,
        well_row: Integer,
        kind: LimitKind,
        quantity: RateQuantity | EconomicQuantity | None = None,
    ) -> Integer | None:
        """
        Finds a well's limit row matching its kind and, when applicable,
        its quantity.

        :param well_row: The well's row.
        :param kind: The kind of limit to find.
        :param quantity: The quantity to find the limit for.
        :returns: The row, or `None` if this well has no economic limit for `quantity`.
        """
        quantity_tag = None
        if kind == LimitKind.RATE:
            if not isinstance(quantity, RateQuantity):
                return None
            quantity_tag = RATE_QUANTITY_TAG[quantity]
        elif kind == LimitKind.ECONOMIC:
            if not isinstance(quantity, EconomicQuantity):
                return None
            quantity_tag = ECONOMIC_QUANTITY_TAG[quantity]
        elif quantity is not None:
            return None

        for row in self.limit_rows(well_row=well_row):
            if self.kinds[row] != kind:
                continue
            if quantity_tag is None or self.quantities[row] == quantity_tag:
                return row
        return None

    @typing.overload
    def get_quantity(self, *, row: Integer) -> RateQuantity | EconomicQuantity | None: ...
    @typing.overload
    def get_quantity(
        self, *, row: IntArray[OneDimension]
    ) -> list[RateQuantity | EconomicQuantity | None]: ...
    def get_quantity(
        self, *, row: IntOrArray[OneDimension]
    ) -> RateQuantity | EconomicQuantity | list[RateQuantity | EconomicQuantity | None] | None:
        """
        :param row: One limit row, or an array of them.
        :returns: Each row's own quantity (a `RateQuantity` on a `RATE`
            row, an `EconomicQuantity` on an `ECONOMIC` row, `None` on a
            `BHP`/`THP` row). A single value for a single `row`, a list for an array.
        """
        if np.isscalar(row):
            return self.get_quantity_at(row=typing.cast(Integer, row))
        return [self.get_quantity_at(row=one_row) for one_row in np.atleast_1d(row)]

    def get_quantity_at(self, *, row: Integer) -> RateQuantity | EconomicQuantity | None:
        """
        :param row: The limit row to read.
        :returns: This row's own quantity, per `get_quantity`'s own rules.
        """
        kind = LimitKind(self.kinds[row])
        tag = self.quantities[row]
        if tag == UNSET_INT:
            return None
        if kind == LimitKind.RATE:
            return RATE_QUANTITY_FROM_TAG[tag]
        if kind == LimitKind.ECONOMIC:
            return ECONOMIC_QUANTITY_FROM_TAG[tag]
        return None

    @typing.overload
    def get_kind(self, *, row: Integer) -> LimitKind: ...
    @typing.overload
    def get_kind(self, *, row: IntArray[OneDimension]) -> list[LimitKind]: ...
    def get_kind(self, *, row: IntOrArray[OneDimension]) -> LimitKind | list[LimitKind]:
        """
        :param row: One limit row, or an array of them.
        :returns: Each row's own kind. A single `LimitKind` for a single
            `row`, a list for an array.
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            return LimitKind(self.kinds[row])
        return [LimitKind(kind) for kind in self.kinds[row]]  # type: ignore[index]

    @typing.overload
    def get_min_value(self, *, row: Integer) -> Number | None: ...
    @typing.overload
    def get_min_value(self, *, row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_min_value(
        self, *, row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param row: One limit row, or an array of them.
        :returns: Each row's own floor. For a single `row`, a Number or
            `None` if unset. For an array, the raw array (`NaN` means unset).
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            return none_if_nan(self.min_values[row])
        return self.min_values[row]  # type: ignore[index]

    def set_min_value(
        self, *, row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more limit rows' floor in place.

        :param row: One limit row, or an array of them.
        :param value: The new floor. A single value applies to every row
            in `row`; an array sets each row to its own value.
        """
        self.min_values[row] = value

    @typing.overload
    def get_max_value(self, *, row: Integer) -> Number | None: ...
    @typing.overload
    def get_max_value(self, *, row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_max_value(
        self, *, row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param row: One limit row, or an array of them.
        :returns: Each row's own ceiling. For a single `row`, a float or
            `None` if unset. For an array, the raw array (`NaN` means unset).
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            return none_if_nan(self.max_values[row])
        return self.max_values[row]  # type: ignore[index]

    def set_max_value(
        self, *, row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more limit rows' ceiling in place.

        :param row: One limit row, or an array of them.
        :param value: The new ceiling. A single value applies to every
            row in `row`; an array sets each row to its own value.
        """
        self.max_values[row] = value

    @typing.overload
    def get_workover_action(self, *, row: Integer) -> WorkoverAction | None: ...
    @typing.overload
    def get_workover_action(
        self, *, row: IntArray[OneDimension]
    ) -> list[WorkoverAction | None]: ...
    def get_workover_action(
        self, *, row: IntOrArray[OneDimension]
    ) -> WorkoverAction | list[WorkoverAction | None] | None:
        """
        :param row: One limit row, or an array of them.
        :returns: Each `ECONOMIC` row's own workover action, or `None`
            on a non-economic row. A single value for a single `row`, a
            list for an array.
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            tag = self.workover_actions[row]
            return None if tag == UNSET_INT else WORKOVER_ACTION_FROM_TAG[tag]
        return [
            None if tag == UNSET_INT else WORKOVER_ACTION_FROM_TAG[tag]
            for tag in self.workover_actions[row]  # type: ignore[index]
        ]

    def set_workover_action(
        self,
        *,
        row: IntOrArray[OneDimension],
        action: WorkoverAction | typing.Sequence[WorkoverAction],
    ) -> None:
        """
        Overwrites one or more `ECONOMIC` rows' workover action in place.

        :param row: One limit row, or an array of them.
        :param action: The new workover action. A single action applies
            to every row in `row`; a sequence sets each row to its own action.
        """
        if isinstance(action, WorkoverAction):
            self.workover_actions[row] = WORKOVER_ACTION_TAG[action]
        else:
            self.workover_actions[row] = [WORKOVER_ACTION_TAG[one_action] for one_action in action]

    @typing.overload
    def get_end_run(self, *, row: Integer) -> Boolean: ...
    @typing.overload
    def get_end_run(self, *, row: IntArray[OneDimension]) -> list[Boolean]: ...
    def get_end_run(self, *, row: IntOrArray[OneDimension]) -> Boolean | list[Boolean]:
        """
        :param row: One limit row, or an array of them.
        :returns: Whether breaching each row should stop the whole run.
            A single `bool` for a single `row`, a list for an array.
        """
        if np.isscalar(row):
            row = typing.cast(Integer, row)
            return bool(self.end_run_flags[row])
        return [bool(flag) for flag in self.end_run_flags[row]]  # type: ignore[index]

    def set_end_run(
        self, *, row: IntOrArray[OneDimension], end_run: Boolean | typing.Sequence[Boolean]
    ) -> None:
        """
        Overwrites one or more `ECONOMIC` rows' end-run flag in place.

        :param row: One limit row, or an array of them.
        :param end_run: Whether breaching each row should stop the whole
            run. A single `bool` applies to every row in `row`; a
            sequence sets each row to its own flag.
        """
        if isinstance(end_run, bool):
            self.end_run_flags[row] = 1 if end_run else 0
        else:
            self.end_run_flags[row] = [
                1 if one_end_run else 0
                for one_end_run in end_run  # type: ignore[index]
            ]


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

    vfp_table_numbers: IntArray[OneDimension]
    """Shape `(n_wells,)`. Number of each well's VFP table, `UNSET_INT` where none is assigned."""

    artificial_lift_quantities: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Artificial lift quantity each well operates at (`0` = no lift)."""

    limits: CompiledLimits

    @typing.overload
    def get_control_mode(
        self, *, well_row: Integer
    ) -> ProducerControlMode | InjectorControlMode: ...
    @typing.overload
    def get_control_mode(
        self, *, well_row: IntArray[OneDimension]
    ) -> list[ProducerControlMode | InjectorControlMode]: ...
    def get_control_mode(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> (
        ProducerControlMode | InjectorControlMode | list[ProducerControlMode | InjectorControlMode]
    ):
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own control mode. A single mode for a
            single `well_row`, a list for an array.
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            is_injector = self.well_kinds[well_row] == WellKind.INJECTOR
            from_tag = INJECTOR_MODE_FROM_TAG if is_injector else PRODUCER_MODE_FROM_TAG
            return from_tag[self.control_modes[well_row]]
        return [
            (INJECTOR_MODE_FROM_TAG if kind == WellKind.INJECTOR else PRODUCER_MODE_FROM_TAG)[mode]
            for kind, mode in zip(
                self.well_kinds[well_row],  # type: ignore[index]
                self.control_modes[well_row],  # type: ignore[index]
                strict=True,
            )
        ]

    def set_control_mode(
        self,
        *,
        well_row: IntOrArray[OneDimension],
        mode: (
            ProducerControlMode
            | InjectorControlMode
            | typing.Sequence[ProducerControlMode | InjectorControlMode]
        ),
    ) -> None:
        """
        Sets one or more wells' control mode in place. Each `mode` must
        match its own well's kind.

        :param well_row: One well's row, or an array of them.
        :param mode: The new control mode. A single mode applies to
            every row in `well_row`; a sequence sets each row to its own mode.
        """
        if isinstance(well_row, typing.Sequence) or (
            isinstance(well_row, np.ndarray) and well_row.ndim > 0
        ):
            modes = mode if isinstance(mode, typing.Sequence) else [mode] * len(well_row)
            for one_row, one_mode in zip(well_row, modes, strict=True):
                self.set_control_mode(well_row=one_row, mode=one_mode)
            return

        is_injector = self.well_kinds[well_row] == WellKind.INJECTOR
        tag_map = INJECTOR_MODE_TAG if is_injector else PRODUCER_MODE_TAG
        mode = typing.cast(ProducerControlMode | InjectorControlMode, mode)
        self.control_modes[well_row] = tag_map[mode]  # type: ignore[index]

    @typing.overload
    def get_injected_phase(self, *, well_row: Integer) -> FluidPhase | None: ...
    @typing.overload
    def get_injected_phase(
        self, *, well_row: IntArray[OneDimension]
    ) -> list[FluidPhase | None]: ...
    def get_injected_phase(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> FluidPhase | list[FluidPhase | None] | None:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each injector's own injected phase, or `None` on a
            producer or unset injector. A single value for a single
            `well_row`, a list for an array.
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            tag = self.injected_phases[well_row]
            return None if tag == UNSET_INT else FLUID_PHASE_FROM_TAG[tag]
        return [
            None if tag == UNSET_INT else FLUID_PHASE_FROM_TAG[tag]
            for tag in self.injected_phases[well_row]  # type: ignore[index]
        ]

    def set_injected_phase(
        self,
        *,
        well_row: IntOrArray[OneDimension],
        phase: FluidPhase | typing.Sequence[FluidPhase],
    ) -> None:
        """
        Overwrites one or more injectors' injected phase in place.

        :param well_row: One well's row, or an array of them.
        :param phase: The new injected phase. A single phase applies to
            every row in `well_row`; a sequence sets each row to its own phase.
        """
        if isinstance(phase, FluidPhase):
            self.injected_phases[well_row] = FLUID_PHASE_TAG[phase]
        else:
            self.injected_phases[well_row] = [FLUID_PHASE_TAG[one_phase] for one_phase in phase]

    @typing.overload
    def get_target_rate(self, *, well_row: Integer) -> Number | None: ...
    @typing.overload
    def get_target_rate(
        self, *, well_row: IntArray[OneDimension]
    ) -> NumberArray[OneDimension]: ...
    def get_target_rate(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own target rate. For a single `well_row`,
            a float or `None` if unset. For an array, the raw array
            (`NaN` means unset).
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return none_if_nan(self.target_rates[well_row])
        return self.target_rates[well_row]  # type: ignore[index]

    def set_target_rate(
        self, *, well_row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' target rate in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new target rate. A single value applies to
            every row in `well_row`; an array sets each row to its own value.
        """
        self.target_rates[well_row] = value

    @typing.overload
    def get_target_bhp(self, *, well_row: Integer) -> Number | None: ...
    @typing.overload
    def get_target_bhp(self, *, well_row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_target_bhp(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own target BHP. For a single `well_row`,
            a Number or `None` if unset. For an array, the raw array
            (`NaN` means unset).
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return none_if_nan(self.target_bhps[well_row])
        return self.target_bhps[well_row]  # type: ignore[index]

    def set_target_bhp(
        self, *, well_row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' target BHP in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new target BHP. A single value applies to
            every row in `well_row`; an array sets each row to its own value.
        """
        self.target_bhps[well_row] = value

    @typing.overload
    def get_target_thp(self, *, well_row: Integer) -> Number | None: ...
    @typing.overload
    def get_target_thp(self, *, well_row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_target_thp(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own target THP. For a single `well_row`,
            a Number or `None` if unset. For an array, the raw array
            (`NaN` means unset).
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return none_if_nan(self.target_thps[well_row])
        return self.target_thps[well_row]  # type: ignore[index]

    def set_target_thp(
        self, *, well_row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' target THP in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new target THP. A single value applies to
            every row in `well_row`; an array sets each row to its own value.
        """
        self.target_thps[well_row] = value

    @typing.overload
    def get_efficiency_factor(self, *, well_row: Integer) -> Number: ...
    @typing.overload
    def get_efficiency_factor(
        self, *, well_row: IntArray[OneDimension]
    ) -> NumberArray[OneDimension]: ...
    def get_efficiency_factor(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> NumberOrArray[OneDimension]:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own efficiency factor. Matches `well_row`'s own shape.
        """
        return self.efficiency_factors[well_row]  # type: ignore[index]

    def set_efficiency_factor(
        self, *, well_row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' efficiency factor in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new efficiency factor. A single value applies
            to every row in `well_row`; an array sets each row to its own value.
        """
        self.efficiency_factors[well_row] = value

    @typing.overload
    def get_guide_rate(self, *, well_row: Integer) -> Number | None: ...
    @typing.overload
    def get_guide_rate(self, *, well_row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_guide_rate(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own guide rate. For a single `well_row`, a
            Number or `None` if unset. For an array, the raw array (`NaN` means unset).
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return none_if_nan(self.guide_rates[well_row])
        return self.guide_rates[well_row]  # type: ignore[index]

    def set_guide_rate(
        self, *, well_row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' guide rate in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new guide rate. A single value applies to
            every row in `well_row`; an array sets each row to its own value.
        """
        self.guide_rates[well_row] = value

    @typing.overload
    def get_vfp_table_number(self, *, well_row: Integer) -> Integer | None: ...
    @typing.overload
    def get_vfp_table_number(
        self, *, well_row: IntArray[OneDimension]
    ) -> IntArray[OneDimension]: ...
    def get_vfp_table_number(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> Integer | IntArray[OneDimension] | None:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's VFP table number. For a single `well_row`, an int or `None` if no
            table is assigned. For an array, the raw array (`UNSET_INT` means none).
        """
        if np.isscalar(well_row):
            number = self.vfp_table_numbers[typing.cast(Integer, well_row)]
            return None if number == UNSET_INT else int(number)
        return self.vfp_table_numbers[well_row]  # type: ignore[index]

    def set_vfp_table_number(
        self, *, well_row: IntOrArray[OneDimension], value: IntOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' VFP table number in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new table number (`UNSET_INT` for none). A single value applies to
            every row in `well_row`; an array sets each row to its own value.
        """
        self.vfp_table_numbers[well_row] = value

    @typing.overload
    def get_artificial_lift_quantity(self, *, well_row: Integer) -> Number: ...
    @typing.overload
    def get_artificial_lift_quantity(
        self, *, well_row: IntArray[OneDimension]
    ) -> NumberArray[OneDimension]: ...
    def get_artificial_lift_quantity(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> NumberOrArray[OneDimension]:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's artificial lift quantity. Matches `well_row`'s own shape.
        """
        return self.artificial_lift_quantities[well_row]  # type: ignore[index]

    def set_artificial_lift_quantity(
        self, *, well_row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' artificial lift quantity in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new artificial lift quantity. A single value applies to every row in
            `well_row`; an array sets each row to its own value.
        """
        self.artificial_lift_quantities[well_row] = value


class CompiledGroupLimits(typing.NamedTuple):
    """
    Every group's own economic limits (deck `GECON`), flattened
    row-per-limit and CSR-indexed by group. A group with no `GECON`
    limits has an empty row range. Every row is `ECONOMIC` kind - a group
    never carries a `BHP`/`THP`/`RATE` limit, only a well does.
    """

    group_offsets: IntArray[OneDimension]
    """Shape `(n_groups + 1,)`."""

    kinds: IntArray[OneDimension]
    """Shape `(n_rows,)`. Always `LimitKind.ECONOMIC`."""

    quantities: IntArray[OneDimension]
    """Shape `(n_rows,)`. An `EconomicQuantityTag` for each row."""

    min_values: NumberArray[OneDimension]
    """Shape `(n_rows,)`. `NaN` where this row has no floor."""

    max_values: NumberArray[OneDimension]
    """Shape `(n_rows,)`. `NaN` where this row has no ceiling."""

    workover_actions: IntArray[OneDimension]
    """Shape `(n_rows,)`. A `WorkoverActionTag` for each row."""

    end_run_flags: IntArray[OneDimension]
    """Shape `(n_rows,)`. `1` on a row that should stop the whole run once breached, `0` otherwise."""

    def limit_rows(self, *, group_row: Integer) -> range:
        """
        A group's own limit-row range, without the caller needing to know about `group_offsets`.

        :param group_row: The group's row in `CompiledGroupControls.names`.
        :returns: `range(start, end)` over this group's limit rows.
        """
        return range(self.group_offsets[group_row], self.group_offsets[group_row + 1])

    def find_limit_row(
        self, *, group_row: Integer, quantity: "EconomicQuantity"
    ) -> Integer | None:
        """
        Finds a group's limit row matching `quantity`. Every row here is
        `ECONOMIC`, so there's no `kind` to match on the way
        `CompiledLimits.find_limit_row` does for a well.

        :param group_row: The group's row.
        :param quantity: The quantity to find the limit for.
        :returns: The row, or `None` if this group has no limit for `quantity`.
        """
        quantity_tag = ECONOMIC_QUANTITY_TAG[quantity]
        for row in self.limit_rows(group_row=group_row):
            if self.quantities[row] == quantity_tag:
                return row
        return None

    def set_min_value(
        self, *, row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more limit rows' floor in place.

        :param row: One limit row, or an array of them.
        :param value: The new floor. A single value applies to every row
            in `row`; an array sets each row to its own value.
        """
        self.min_values[row] = value

    def set_max_value(
        self, *, row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more limit rows' ceiling in place.

        :param row: One limit row, or an array of them.
        :param value: The new ceiling. A single value applies to every
            row in `row`; an array sets each row to its own value.
        """
        self.max_values[row] = value

    def set_workover_action(
        self,
        *,
        row: IntOrArray[OneDimension],
        action: WorkoverAction | typing.Sequence[WorkoverAction],
    ) -> None:
        """
        Overwrites one or more limit rows' workover action in place.

        :param row: One limit row, or an array of them.
        :param action: The new workover action. A single action applies
            to every row in `row`; a sequence sets each row to its own action.
        """
        if isinstance(action, WorkoverAction):
            self.workover_actions[row] = WORKOVER_ACTION_TAG[action]
        else:
            self.workover_actions[row] = [WORKOVER_ACTION_TAG[one_action] for one_action in action]

    def set_end_run(
        self, *, row: IntOrArray[OneDimension], end_run: Boolean | typing.Sequence[Boolean]
    ) -> None:
        """
        Overwrites one or more limit rows' end-run flag in place.

        :param row: One limit row, or an array of them.
        :param end_run: Whether breaching each row should stop the whole
            run. A single `bool` applies to every row in `row`; a
            sequence sets each row to its own flag.
        """
        if isinstance(end_run, bool):
            self.end_run_flags[row] = 1 if end_run else 0
        else:
            self.end_run_flags[row] = [1 if one_end_run else 0 for one_end_run in end_run]  # type: ignore


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

    limits: CompiledGroupLimits
    """This group's own `GECON` economic limits, one row per group even when empty."""

    @typing.overload
    def group_row(self, name: str) -> Integer: ...  # type: ignore
    @typing.overload
    def group_row(self, name: typing.Iterable[str]) -> list[Integer]: ...
    def group_row(self, name: str | typing.Iterable[str]) -> Integer | list[Integer]:
        """
        A group's row, by name.

        :param name: The group to find, or a sequence of them.
        :returns: The group's row for a single `name`, a list for a sequence.
        :raises ValidationError: If any named group isn't in this system.
        """
        if isinstance(name, str):
            try:
                return name_to_row_map(self.names)[name]
            except KeyError:
                raise ValidationError(
                    f"No group named {name!r} in this compiled system."
                ) from None
        return [self.group_row(name=one_name) for one_name in name]


class CompiledWellSystem(typing.NamedTuple):
    """Top-level compiled wells bundle for the solver hot path."""

    names: tuple[str, ...]
    """Shape `(n_wells,)`. Every other array here is positional against this order."""

    well_kinds: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. Duplicated from `controls.well_kinds`, kept in
    lockstep by construction, for a kernel that needs well identity
    without threading `controls` through too.
    """

    schedule_statuses: IntArray[OneDimension]
    """Shape `(n_wells,)`. `1` for `WellStatus.ACTIVE`, `0` for `PENDING`."""

    open_statuses: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. `1` for a well whose `Well.open_status` is `OPEN`, `0` for `SHUT`.
    A shut well flows nothing, whatever the status of its connections, which are left as they
    are for when the well is opened again.
    """

    reference_depths: NumberArray[OneDimension]
    """Shape `(n_wells,)`."""

    reference_measured_depths: NumberArray[OneDimension]
    """
    Shape `(n_wells,)`. Measured depth along the wellbore of each well's reference depth. Equal
    to the reference depth for a vertical well; for a well with a trajectory, the shallowest
    point along it at that depth.
    """

    d_factors: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Non-Darcy flow coefficient. `NaN` where unset."""

    perforations: CompiledPerforations
    """Compiled perforation data for the solver hot path."""

    controls: CompiledWellControls
    """Compiled well control data aligned with `names` and `well_kinds`."""

    group_controls: CompiledGroupControls | None
    """Compiled group-control data, or `None` when no group controls exist."""

    unit_system: UnitSystem
    """Unit system used to interpret the compiled well data."""

    @typing.overload
    def well_row(self, name: str) -> Integer: ...  # type: ignore
    @typing.overload
    def well_row(self, name: typing.Iterable[str]) -> list[Integer]: ...
    def well_row(self, name: str | typing.Iterable[str]) -> Integer | list[Integer]:
        """
        A well's row, by name.

        :param name: The well to find, or a sequence of them.
        :returns: The well's row for a single `name`, a list for a sequence.
        :raises ValidationError: If any named well isn't in this system.
        """
        if isinstance(name, str):
            try:
                return name_to_row_map(self.names)[name]
            except KeyError:
                raise ValidationError(f"No well named {name!r} in this compiled system.") from None
        return [self.well_row(name=one_name) for one_name in name]

    def has_well(self, name: str) -> bool:
        """
        :param name: The well to check for.
        :returns: Whether a well with this name exists in this system.
        """
        return name in self.names

    def is_injector(self, well: str | Integer) -> bool:
        """Whether `well`, given by name or row, is an injector."""
        well_row = self.well_row(name=well) if isinstance(well, str) else well
        return self.well_kinds[well_row] == WellKind.INJECTOR

    def is_producer(self, well: str | Integer) -> bool:
        """Whether `well`, given by name or row, is a producer."""
        well_row = self.well_row(name=well) if isinstance(well, str) else well
        return self.well_kinds[well_row] == WellKind.PRODUCER

    @typing.overload
    def get_producers(self, *, row: typing.Literal[False] = False) -> tuple[str, ...]: ...
    @typing.overload
    def get_producers(self, *, row: typing.Literal[True]) -> tuple[Integer, ...]: ...
    def get_producers(self, *, row: bool = False) -> tuple[str, ...] | tuple[Integer, ...]:
        """Returns producer names, or their row indices when `row` is `True`."""
        if row:
            return tuple(
                index for index, kind in enumerate(self.well_kinds) if kind == WellKind.PRODUCER
            )
        return tuple(
            name
            for name, kind in zip(self.names, self.well_kinds, strict=True)
            if kind == WellKind.PRODUCER
        )

    @typing.overload
    def get_injectors(self, *, row: typing.Literal[False] = False) -> tuple[str, ...]: ...
    @typing.overload
    def get_injectors(self, *, row: typing.Literal[True]) -> tuple[Integer, ...]: ...
    def get_injectors(self, *, row: bool = False) -> tuple[str, ...] | tuple[Integer, ...]:
        """Returns injector names, or their row indices when `row` is `True`."""
        if row:
            return tuple(
                index for index, kind in enumerate(self.well_kinds) if kind == WellKind.INJECTOR
            )
        return tuple(
            name
            for name, kind in zip(self.names, self.well_kinds, strict=True)
            if kind == WellKind.INJECTOR
        )

    @typing.overload
    def get_well_type(self, *, well_row: Integer) -> WellType: ...
    @typing.overload
    def get_well_type(self, *, well_row: IntArray[OneDimension]) -> list[WellType]: ...
    def get_well_type(self, *, well_row: IntOrArray[OneDimension]) -> WellType | list[WellType]:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own type. A single `WellType` for a single
            `well_row`, a list for an array.
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return (
                WellType.PRODUCER
                if self.well_kinds[well_row] == WellKind.PRODUCER
                else WellType.INJECTOR
            )
        return [
            WellType.PRODUCER if kind == WellKind.PRODUCER else WellType.INJECTOR
            for kind in self.well_kinds[well_row]  # type: ignore[index]
        ]

    @typing.overload
    def get_schedule_status(self, *, well_row: Integer) -> WellStatus: ...
    @typing.overload
    def get_schedule_status(self, *, well_row: IntArray[OneDimension]) -> list[WellStatus]: ...
    def get_schedule_status(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> WellStatus | list[WellStatus]:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own active/pending status. A single
            `WellStatus` for a single `well_row`, a list for an array.
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return WellStatus.ACTIVE if self.schedule_statuses[well_row] else WellStatus.PENDING
        return [
            WellStatus.ACTIVE if tag else WellStatus.PENDING
            for tag in self.schedule_statuses[well_row]  # type: ignore[index]
        ]

    def set_schedule_status(
        self,
        *,
        well_row: IntOrArray[OneDimension],
        status: WellStatus | typing.Sequence[WellStatus],
    ) -> None:
        """
        Sets one or more wells' active/pending status in place.

        :param well_row: One well's row, or an array of them.
        :param status: The new active/pending status. A single `WellStatus`
            applies to every row in `well_row`; a sequence sets each row to its own status.
        """
        if isinstance(status, WellStatus):
            self.schedule_statuses[well_row] = get_well_status_tag(status=status)
        else:
            self.schedule_statuses[well_row] = [
                get_well_status_tag(status=one_status) for one_status in status
            ]

    @typing.overload
    def get_open_status(self, *, well_row: Integer) -> CompletionStatus: ...
    @typing.overload
    def get_open_status(self, *, well_row: IntArray[OneDimension]) -> list[CompletionStatus]: ...
    def get_open_status(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> CompletionStatus | list[CompletionStatus]:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own open or shut status. A single `CompletionStatus` for a
            single `well_row`, a list for an array.
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return CompletionStatus.OPEN if self.open_statuses[well_row] else CompletionStatus.SHUT
        return [
            CompletionStatus.OPEN if tag else CompletionStatus.SHUT
            for tag in self.open_statuses[well_row]  # type: ignore[index]
        ]

    def set_open_status(
        self,
        *,
        well_row: IntOrArray[OneDimension],
        status: CompletionStatus | typing.Sequence[CompletionStatus],
    ) -> None:
        """
        Sets one or more wells' open or shut status in place. The status of each well's
        connections is left as it is.

        :param well_row: One well's row, or an array of them.
        :param status: The new open or shut status. A single `CompletionStatus` applies to
            every row in `well_row`; a sequence sets each row to its own status.
        """
        if isinstance(status, CompletionStatus):
            self.open_statuses[well_row] = get_completion_status_tag(status=status)
        else:
            self.open_statuses[well_row] = [
                get_completion_status_tag(status=one_status) for one_status in status
            ]

    def get_flowing_connection_rows(self, *, well_row: Integer) -> IntArray[OneDimension]:
        """
        The connections of a well that can flow right now: in effect, open, and on a well that
        is itself in effect and open. This is the one place that decides it, so whatever
        supplies a well's connection fluids must supply them for exactly these rows, in order.

        :param well_row: The well's row in `names`.
        :returns: The rows of the well's flowing connections, in order. Empty if the well is
            pending or shut, or has no connection that is open and in effect.
        """
        if self.schedule_statuses[well_row] == 0 or self.open_statuses[well_row] == 0:
            return np.empty(0, dtype=np.intp)  # type: ignore[return-value]

        perforations = self.perforations
        connection_rows = perforations.connection_rows(well_row=well_row)
        rows = np.arange(connection_rows.start, connection_rows.stop)
        flowing = (perforations.completion_statuses[rows] == 1) & (
            perforations.schedule_statuses[rows] == 1
        )
        return rows[flowing]  # type: ignore[return-value]

    @typing.overload
    def get_reference_depth(self, *, well_row: Integer) -> Number: ...
    @typing.overload
    def get_reference_depth(
        self, *, well_row: IntArray[OneDimension]
    ) -> NumberArray[OneDimension]: ...
    def get_reference_depth(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> NumberOrArray[OneDimension]:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own reference depth. Matches `well_row`'s own shape.
        """
        return self.reference_depths[well_row]  # type: ignore[index]

    @typing.overload
    def get_d_factor(self, *, well_row: Integer) -> Number | None: ...
    @typing.overload
    def get_d_factor(self, *, well_row: IntArray[OneDimension]) -> NumberArray[OneDimension]: ...
    def get_d_factor(
        self, *, well_row: IntOrArray[OneDimension]
    ) -> Number | NumberArray[OneDimension] | None:
        """
        :param well_row: One well's row, or an array of them.
        :returns: Each well's own non-Darcy coefficient. For a single
            `well_row`, a float or `None` if unset. For an array, the
            raw array (`NaN` means unset).
        """
        if np.isscalar(well_row):
            well_row = typing.cast(Integer, well_row)
            return none_if_nan(self.d_factors[well_row])
        return self.d_factors[well_row]  # type: ignore[index]

    def set_d_factor(
        self, *, well_row: IntOrArray[OneDimension], value: NumberOrArray[OneDimension]
    ) -> None:
        """
        Overwrites one or more wells' non-Darcy coefficient in place.

        :param well_row: One well's row, or an array of them.
        :param value: The new coefficient. A single value applies to
            every row in `well_row`; an array sets each row to its own value.
        """
        self.d_factors[well_row] = value


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


def to_int_array(
    values: typing.Iterable[Integer], *, dtype: npt.DTypeLike = np.int32
) -> IntArray[OneDimension]:
    """
    Packs integer values into a one-dimensional array.

    :param values: The values to pack.
    :param dtype: Integer dtype of the result.
    :returns: A one-dimensional integer array.
    """
    return typing.cast(IntArray[OneDimension], np.fromiter(values, dtype=dtype))


def to_number_array(
    values: typing.Iterable[Number], *, dtype: npt.DTypeLike
) -> NumberArray[OneDimension]:
    """
    Packs numeric values into a one-dimensional array.

    :param values: The values to pack.
    :param dtype: Floating-point dtype of the result.
    :returns: A one-dimensional floating-point array.
    """
    return typing.cast(NumberArray[OneDimension], np.fromiter(values, dtype=dtype))


class LimitColumns:
    """
    Flattens the limits of a sequence of wells or groups into parallel columns,
    one row per limit, with offsets marking where each well's or group's rows end.
    """

    __slots__ = (
        "end_run_flags",
        "kinds",
        "max_values",
        "min_values",
        "offsets",
        "quantities",
        "workover_actions",
    )

    def __init__(self) -> None:
        self.offsets: list[Integer] = [0]
        """Cumulative row count after each well or group added so far, starting at `0`."""

        self.kinds: list[Integer] = []
        """`LimitKind` tag of each limit."""

        self.quantities: list[Integer] = []
        """Rate or economic quantity tag of each limit, `UNSET_INT` for BHP and THP limits."""

        self.min_values: list[Number] = []
        """Lower bound of each limit, `NaN` where absent."""

        self.max_values: list[Number] = []
        """Upper bound of each limit, `NaN` where absent."""

        self.workover_actions: list[Integer] = []
        """Workover action tag of each limit, `UNSET_INT` for non-economic limits."""

        self.end_run_flags: list[Integer] = []
        """`1` for a limit that ends the run when breached, otherwise `0`."""

    def extend(self, limits: typing.Iterable[Limit]) -> None:
        """
        Adds one well's or group's limits as the next block of rows.

        :param limits: Limits to add. An empty sequence adds a block with no rows.
        :raises ValidationError: If `limits` contains an unrecognized `Limit` subtype.
        """
        kinds = self.kinds
        quantities = self.quantities
        min_values = self.min_values
        max_values = self.max_values
        workover_actions = self.workover_actions
        end_run_flags = self.end_run_flags

        for limit in limits:
            if isinstance(limit, (BHPLimit, THPLimit)):
                kinds.append(LimitKind.BHP if isinstance(limit, BHPLimit) else LimitKind.THP)
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
        self.offsets.append(len(kinds))

    def to_arrays(
        self, *, dtype: npt.DTypeLike
    ) -> tuple[
        IntArray[OneDimension],
        IntArray[OneDimension],
        IntArray[OneDimension],
        NumberArray[OneDimension],
        NumberArray[OneDimension],
        IntArray[OneDimension],
        IntArray[OneDimension],
    ]:
        """
        Packs the columns into arrays.

        :param dtype: Floating-point dtype of the bound columns.
        :returns: `(offsets, kinds, quantities, min_values, max_values,
            workover_actions, end_run_flags)`.
        """
        return (
            to_int_array(self.offsets, dtype=np.int64),
            to_int_array(self.kinds),
            to_int_array(self.quantities),
            to_number_array(self.min_values, dtype=dtype),
            to_number_array(self.max_values, dtype=dtype),
            to_int_array(self.workover_actions),
            to_int_array(self.end_run_flags),
        )


def compile_perforations(
    names: typing.Sequence[str],
    wells: Wells,
    *,
    grid: Grid,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    horizontal_tolerance: Number | None = None,
    intersection_method: GridIntersectionMethod = "aabb",
    search_radius: Number | None = None,
    dtype: npt.DTypeLike = None,
) -> CompiledPerforations:
    """
    Builds `CompiledPerforations` for a set of wells, in order.

    Every perforation of every well is resolved, open or shut, and each
    connection's open/shut status is read from its own perforation.

    :param names: Well names, in the row order `CompiledWellSystem` will use.
    :param wells: Source `Wells`.
    :param grid: Grid to resolve against.
    :param permeabilities: Forwarded to `build_well_index`.
    :param horizontal_tolerance: Forwarded to `build_well_index`.
    :param intersection_method: Forwarded to `build_well_index`.
    :param search_radius: Forwarded to `build_well_index`.
    :param dtype: Floating-point dtype of the compiled arrays.
    :returns: `CompiledPerforations`, one row per (well, perforation, cell) triple.
    :raises ValidationError: If `grid.unit_system` doesn't match `wells.unit_system`.
    """
    if grid.unit_system != wells.unit_system:
        raise ValidationError(
            f"`Grid.unit_system` ({grid.unit_system.value}) != `Wells."
            f"unit_system` ({wells.unit_system.value})."
        )

    well_offsets = [0]
    connections: list[PerforationIndex] = []
    ordinals: list[Integer] = []
    for name in names:
        well = wells[name]
        well_perforations = well.perforations
        ordinal_by_id = {
            id(perforation): ordinal for ordinal, perforation in enumerate(well_perforations)
        }
        well_connections = build_well_index(
            grid=grid,
            well=well,
            permeabilities=permeabilities,
            horizontal_tolerance=horizontal_tolerance,
            intersection_method=intersection_method,
            search_radius=search_radius,
            perforations=well_perforations,
        ).perforations
        connections.extend(well_connections)
        ordinals.extend(
            ordinal_by_id[id(connection.perforation)] for connection in well_connections
        )
        well_offsets.append(len(connections))

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    count = len(connections)

    def number_column(values: typing.Iterable[Number]) -> NumberArray[OneDimension]:
        return typing.cast(
            NumberArray[OneDimension], np.fromiter(values, dtype=dtype, count=count)
        )

    def int_column(values: typing.Iterable[Integer]) -> IntArray[OneDimension]:
        return typing.cast(
            IntArray[OneDimension], np.fromiter(values, dtype=np.int32, count=count)
        )

    return CompiledPerforations(
        well_offsets=to_int_array(well_offsets, dtype=np.int64),
        cell_indices=typing.cast(
            IntArray[OneDimension],
            np.fromiter((c.cell_index for c in connections), dtype=np.int64, count=count),
        ),
        perforation_indices=to_int_array(ordinals, dtype=np.int64),
        well_indices=number_column(
            c.well_index if c.well_index is not None else np.nan for c in connections
        ),
        connection_conductivities=number_column(
            c.conductivity if c.conductivity is not None else np.nan for c in connections
        ),
        wellbore_radii=number_column(c.perforation.wellbore_radius for c in connections),
        skins=number_column(c.perforation.skin for c in connections),
        partial_penetration_fractions=number_column(
            c.partial_penetration_fraction for c in connections
        ),
        representative_depths=number_column(c.representative_depth for c in connections),
        representative_measured_depths=number_column(
            c.representative_measured_depth for c in connections
        ),
        inclinations_from_vertical=number_column(c.inclination_from_vertical for c in connections),
        completion_statuses=int_column(
            get_completion_status_tag(status=c.perforation.status) for c in connections
        ),
        schedule_statuses=int_column(
            get_well_status_tag(status=c.perforation.schedule_status) for c in connections
        ),
        saturation_regions=int_column(
            c.perforation.saturation_region
            if c.perforation.saturation_region is not None
            else UNSET_INT
            for c in connections
        ),
    )


def compile_well_controls(
    *,
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
    :param dtype: Floating-point dtype of the compiled arrays.
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
    vfp_table_numbers: list[Integer] = []
    artificial_lift_quantities: list[Number] = []
    limit_columns = LimitColumns()

    for name in names:
        control = controls.get(name)
        kind = (
            WellKind.INJECTOR if wells[name].well_type is WellType.INJECTOR else WellKind.PRODUCER
        )
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
            vfp_table_numbers.append(UNSET_INT)
            artificial_lift_quantities.append(0.0)
            limit_columns.extend(())
            continue

        if isinstance(control, InjectorControl):
            control_modes.append(INJECTOR_MODE_TAG[control.mode])
            injected_phases.append(FLUID_PHASE_TAG[control.injected_phase])
        elif isinstance(control, ProducerControl):
            control_modes.append(PRODUCER_MODE_TAG[control.mode])
            injected_phases.append(UNSET_INT)
        else:
            raise ValidationError(f"Unknown WellControl type: {type(control)!r}.")

        target_bhp = getattr(control, "target_bhp", None)
        target_thp = getattr(control, "target_thp", None)
        target_rate = getattr(control, "target_rate", None)
        guide_rate = getattr(control, "guide_rate", None)
        vfp_table = getattr(control, "vfp_table", None)
        target_bhps.append(target_bhp if target_bhp is not None else np.nan)
        target_thps.append(target_thp if target_thp is not None else np.nan)
        target_rates.append(target_rate if target_rate is not None else np.nan)
        efficiency_factors.append(getattr(control, "efficiency_factor", 1.0))
        guide_rates.append(guide_rate if guide_rate is not None else np.nan)
        vfp_table_numbers.append(vfp_table if vfp_table is not None else UNSET_INT)
        artificial_lift_quantities.append(getattr(control, "artificial_lift_quantity", 0.0))
        limit_columns.extend(getattr(control, "limits", ()))

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    (
        limits_well_offsets,
        limits_kinds,
        limits_quantities,
        limits_min_values,
        limits_max_values,
        limits_workover_actions,
        limits_end_run_flags,
    ) = limit_columns.to_arrays(dtype=dtype)
    return CompiledWellControls(
        well_kinds=to_int_array(well_kinds),
        control_modes=to_int_array(control_modes),
        injected_phases=to_int_array(injected_phases),
        target_rates=to_number_array(target_rates, dtype=dtype),
        target_bhps=to_number_array(target_bhps, dtype=dtype),
        target_thps=to_number_array(target_thps, dtype=dtype),
        efficiency_factors=to_number_array(efficiency_factors, dtype=dtype),
        guide_rates=to_number_array(guide_rates, dtype=dtype),
        vfp_table_numbers=to_int_array(vfp_table_numbers),
        artificial_lift_quantities=to_number_array(artificial_lift_quantities, dtype=dtype),
        limits=CompiledLimits(
            well_offsets=limits_well_offsets,
            kinds=limits_kinds,
            quantities=limits_quantities,
            min_values=limits_min_values,
            max_values=limits_max_values,
            workover_actions=limits_workover_actions,
            end_run_flags=limits_end_run_flags,
        ),
    )


def compile_group_controls(
    group_controls: GroupControls | None,
    *,
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
    :param dtype: Floating-point dtype of the compiled arrays.
    :returns: `CompiledGroupControls`, one row per group with a control,
        in sorted name order; `None` if `group_controls` is `None`. Each
        row's own `limits` covers that group's `GroupControl.limits`
        (deck `GECON`), an empty row range if it has none.
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

    rows_by_group: dict[str | None, list[Integer]] = {}
    for row, well_name in enumerate(names):
        rows_by_group.setdefault(wells[well_name].group, []).append(row)
    member_offsets: list[Integer] = [0]
    member_well_indices: list[Integer] = []
    limit_columns = LimitColumns()

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
            raise ValidationError(f"Unknown `GroupControl` mode type: {type(control.mode)!r}.")
        target_rates.append(control.target_rate if control.target_rate is not None else np.nan)

        if groups is None:
            member_well_indices.extend(rows_by_group.get(name, ()))
        else:
            member_well_indices.extend(
                sorted(
                    row
                    for group_name in (name, *groups.descendants(name))
                    for row in rows_by_group.get(group_name, ())
                )
            )
        member_offsets.append(len(member_well_indices))
        limit_columns.extend(control.limits)

    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    (
        group_limits_offsets,
        group_limits_kinds,
        group_limits_quantities,
        group_limits_min_values,
        group_limits_max_values,
        group_limits_workover_actions,
        group_limits_end_run_flags,
    ) = limit_columns.to_arrays(dtype=dtype)
    return CompiledGroupControls(
        names=tuple(control_names),
        group_kinds=to_int_array(group_kinds),
        control_modes=to_int_array(control_modes),
        injected_phases=to_int_array(injected_phases),
        target_rates=to_number_array(target_rates, dtype=dtype),
        member_offsets=to_int_array(member_offsets),
        member_well_indices=to_int_array(member_well_indices),
        limits=CompiledGroupLimits(
            group_offsets=group_limits_offsets,
            kinds=group_limits_kinds,
            quantities=group_limits_quantities,
            min_values=group_limits_min_values,
            max_values=group_limits_max_values,
            workover_actions=group_limits_workover_actions,
            end_run_flags=group_limits_end_run_flags,
        ),
    )


def compile_well_system(
    wells: Wells,
    controls: WellControls,
    grid: Grid,
    *,
    permeabilities: typing.Mapping[Orientation, NumberArray[OneDimension]],
    group_controls: GroupControls | None = None,
    groups: WellGroups | None = None,
    dtype: npt.DTypeLike = None,
    horizontal_tolerance: Number | None = None,
    intersection_method: GridIntersectionMethod = "aabb",
    search_radius: Number | None = None,
) -> CompiledWellSystem:
    """
    Compiles a rich `Wells`/`WellControls` (from a `WellSystem`) pair
    into a `CompiledWellSystem`.

    Every well `wells` contains is included, regardless of `schedule_status`
    as this is the load-once roster. A `WellStatus.PENDING` well is compiled
    the same as an `ACTIVE` one, just tagged, so a later schedule event can
    activate it in place without recompiling.

    :param wells: Every well that will ever exist across the run.
    :param controls: Current controls. A well absent from `controls`
        compiles to an `UNSET`-mode row (see `compile_well_controls`).
    :param grid: Grid to resolve perforation geometry against.
    :param permeabilities: Per-axis permeability arrays, forwarded to
        `compile_perforations`.
    :param group_controls: Current group controls, if any.
    :param groups: The group hierarchy, if any - `WellSystem.groups`.
        Forwarded to `compile_group_controls` for descendant-aware
        membership; a group control without a hierarchy still compiles,
        with membership restricted to that group's direct members.
    :param dtype: Floating-point dtype of the compiled arrays.
    :param horizontal_tolerance: Forwarded to `compile_perforations`.
    :param intersection_method: Forwarded to `compile_perforations`.
    :param search_radius: Forwarded to `compile_perforations`.
    :returns: `CompiledWellSystem`, well rows in `wells.names` order.
    :raises ValidationError: If `controls.unit_system` doesn't match
        `wells.unit_system`.
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
        horizontal_tolerance=horizontal_tolerance,
        intersection_method=intersection_method,
        search_radius=search_radius,
        dtype=dtype,
    )
    well_controls = compile_well_controls(
        names=names,
        controls=controls,
        wells=wells,
        dtype=dtype,
    )

    roster = [wells[name] for name in names]
    return CompiledWellSystem(
        names=tuple(names),
        well_kinds=well_controls.well_kinds,
        schedule_statuses=to_int_array(
            get_well_status_tag(status=well.schedule_status) for well in roster
        ),
        open_statuses=to_int_array(
            get_completion_status_tag(status=well.open_status) for well in roster
        ),
        reference_depths=to_number_array((well.reference_depth for well in roster), dtype=dtype),
        reference_measured_depths=to_number_array(
            (
                well.trajectory.get_measured_depth_at(well.reference_depth)
                if well.trajectory is not None
                else well.reference_depth
                for well in roster
            ),
            dtype=dtype,
        ),
        d_factors=to_number_array(
            (well.d_factor if well.d_factor is not None else np.nan for well in roster),
            dtype=dtype,
        ),
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
