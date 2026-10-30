"""Per-timestep dynamic well state."""

import enum
import fnmatch
import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.dataframes import (
    AnyDataFrame,
    Backend,
    Column,
    build_dataframe,
    get_grid_columns,
    get_label_column,
    resolve_backend,
    resolve_times,
)
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.stores import StoreSerializable
from bores.types import Boolean, Integer, Number, PhaseValues, UnitConversionTable, UnitSystem
from bores.units import get_conversion_factors, resolve_unit_conversion_table
from bores.utils import scale
from bores.wells.base import Perforation
from bores.wells.controls import InjectorControl, Limit, ProducerControl, WellControl

__all__ = [
    "ConnectionSample",
    "PerforationState",
    "WellState",
    "WellStates",
    "as_dataframe",
]


class ConnectionSample(typing.NamedTuple):
    """Reservoir conditions at one connected cell."""

    cell_index: int
    """Index of the connected grid cell."""

    pressure: Number
    """Reservoir pressure at the connected cell."""

    temperature: Number
    """Reservoir temperature at the connected cell."""

    phase_saturations: PhaseValues
    """Saturation of each phase at the connected cell."""

    phase_mobilities: PhaseValues
    """Mobility of each phase at the connected cell."""

    phase_densities: PhaseValues
    """Density of each phase at the connected cell."""

    phase_viscosities: PhaseValues
    """Viscosity of each phase at the connected cell."""

    gas_liquid_surface_tension: Number
    """Surface tension between the gas and liquid phases."""

    phase_fvfs: PhaseValues
    """
    Reservoir volume per surface volume of each phase, at this
    connection's current pressure.
    """

    solution_gas_oil_ratio: Number = 0.0
    """
    Gas the oil at this connected cell holds in solution, per stock-tank barrel of oil. Needed
    to count the dissolved gas of a well's produced oil, when the tubing head pressure is found
    from a table of tubing fluids. Zero if the oil is taken to hold no gas.
    """

    vaporized_oil_gas_ratio: Number = 0.0
    """
    Oil the gas at this connected cell carries as vapor, in stock-tank barrels per standard cubic
    foot of gas. Zero for dry gas.
    """


@attrs.frozen(kw_only=True, slots=True)
class PerforationState(StoreSerializable):
    """Per-perforation dynamic snapshot for one timestep."""

    perforation: Perforation
    """The perforation this snapshot is for."""

    cell_index: Integer
    """The grid cell this perforation connects to."""

    flowing_pressure: Number
    """Flowing bottomhole pressure at this perforation."""

    rates: PhaseValues
    """Rate of each phase at this perforation, at reservoir conditions."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system this snapshot's dimensioned fields are expressed in."""

    def __attrs_post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts this snapshot to a different unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: This snapshot, with `flowing_pressure` and every phase-rate
            value (reservoir- and surface-condition) converted to `target`.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        rate_factor = factors["reservoir_rate"]
        return attrs.evolve(
            self,
            flowing_pressure=scale(self.flowing_pressure, factors["pressure"]),
            rates=PhaseValues(
                oil=scale(self.rates.oil, rate_factor),
                water=scale(self.rates.water, rate_factor),
                gas=scale(self.rates.gas, rate_factor),
            ),
            unit_system=target,
        )


@attrs.frozen(kw_only=True, slots=True)
class WellState(StoreSerializable):
    """Per-well dynamic snapshot for one timestep."""

    well_name: str
    """Name of the well this snapshot is for."""

    is_open: bool
    """Whether the well is currently flowing."""

    active_control: WellControl
    """The control currently governing this well."""

    bhp: Number
    """Current bottomhole pressure."""

    perforation_states: tuple[PerforationState, ...]
    """Snapshot for each of this well's open perforations. Empty if `is_open` is `False`."""

    rates: PhaseValues
    """Rate of each phase for the whole well, at reservoir conditions."""

    surface_rates: PhaseValues
    """Rate of each phase for the whole well, at surface conditions"""

    active_limit: Limit | None = None
    """The limit currently constraining the well, if any."""

    thp: Number | None = None
    """Current tubing head pressure, if computed."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system this snapshot's dimensioned fields are expressed in."""

    def __attrs_post_init__(self) -> None:
        if not self.well_name:
            raise ValidationError("`well_name` must be a non-empty string.")

        if not self.is_open and self.perforation_states:
            raise ValidationError(
                f"{self.__class__.__name__} for {self.well_name!r} is shut (`is_open=False`) "
                "but has non-empty `perforation_states`; a shut well must "
                "have an empty tuple."
            )

        if self.active_control.unit_system != self.unit_system:
            raise ValidationError(
                f"`active_control.unit_system` ({self.active_control.unit_system.value}) "
                f"!= this {self.__class__.__name__}'s unit_system ({self.unit_system.value})."
            )
        if self.active_limit is not None and self.active_limit.unit_system != self.unit_system:
            raise ValidationError(
                f"`active_limit.unit_system` ({self.active_limit.unit_system.value}) "
                f"!= this {self.__class__.__name__}'s unit_system ({self.unit_system.value})."
            )

    @property
    def total_liquid_rate(self) -> Number:
        """Combined oil and water rate for the well, at reservoir conditions."""
        return self.rates.oil + self.rates.water

    @property
    def total_surface_liquid_rate(self) -> Number:
        """Combined oil and water rate for the well, at surface conditions."""
        return self.surface_rates.oil + self.surface_rates.water

    def get_perforation_state(self, cell_index: int) -> PerforationState:
        """
        Look up and return the first `PerforationState` connected to a given cell.

        :param cell_index: 0-based `Grid` cell index.
        :returns: Matching `PerforationState`.
        :raises KeyError: If no open perforation is connected to `cell_index`.
        """
        for state in self.perforation_states:
            if state.cell_index == cell_index:
                return state
        raise KeyError(
            f"No open perforation connected to `cell_index={cell_index}` for "
            f"well {self.well_name!r}."
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts this snapshot to a different unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: This snapshot, with `bhp`, `thp`, `rates`,
            `surface_rates`, `perforation_states`, `active_control`,
            and `active_limit` (if set) converted to `target`.
        """
        if target == self.unit_system:
            return self
        table = resolve_unit_conversion_table(table)

        factors = get_conversion_factors(self.unit_system, target, table=table)
        reservoir_rate_factor = factors["reservoir_rate"]
        return attrs.evolve(
            self,
            bhp=scale(self.bhp, factors["pressure"]),
            thp=scale(self.thp, factors["pressure"]) if self.thp is not None else None,
            rates=PhaseValues(
                oil=scale(self.rates.oil, reservoir_rate_factor),
                water=scale(self.rates.water, reservoir_rate_factor),
                gas=scale(self.rates.gas, reservoir_rate_factor),
            ),
            surface_rates=PhaseValues(
                oil=scale(self.surface_rates.oil, factors["liquid_surface_rate"]),
                water=scale(self.surface_rates.water, factors["liquid_surface_rate"]),
                gas=scale(self.surface_rates.gas, factors["gas_surface_rate"]),
            ),
            perforation_states=tuple(
                perforation_state.convert(target, table=table)
                for perforation_state in self.perforation_states
            ),
            active_control=self.active_control.convert(target, table=table),
            active_limit=self.active_limit.convert(target, table=table)
            if self.active_limit is not None
            else None,
            unit_system=target,
        )


class WellStates(
    StoreSerializable,
    fields={
        "states": typing.Mapping[str, WellState],
        "unit_system": typing.Optional[UnitSystem],
    },
):
    """Name-keyed collection of `WellState`, one per well, for a single timestep."""

    __slots__ = ("states", "unit_system")

    def __init__(
        self,
        states: typing.Mapping[str, WellState],
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        Initialize a `WellStates` instance.

        :param states: Mapping from well name to `WellState`.
        :param unit_system: Target unit system for every state. None
            requires all states to already share the same unit system.
        :raises ValidationError: If any key doesn't match its value's
            `WellState.well_name`, or (unit_system is None) the states
            don't all share one unit system.
        """
        mismatched = {
            key: state.well_name for key, state in states.items() if key != state.well_name
        }
        if mismatched:
            raise ValidationError(
                f"`states` dict keys must match `WellState.well_name`; "
                f"mismatches (key -> well_name): {mismatched}."
            )

        if unit_system is None:
            systems = {state.unit_system for state in states.values()}
            if len(systems) > 1:
                raise ValidationError(
                    "All states must share the same unit system when "
                    "`unit_system` is not explicitly provided. Found: "
                    f"{sorted(s.value for s in systems)}."
                )
            unit_system = systems.pop() if systems else UnitSystem.FIELD
        else:
            states = {
                name: state if state.unit_system == unit_system else state.convert(unit_system)
                for name, state in states.items()
            }

        self.states = dict(states)
        self.unit_system = unit_system

    def get(self, well_name: str) -> WellState | None:
        """
        Get the `WellState` for a given well.

        :param well_name: Well to look up.
        :returns: WellState, or None if unset.
        """
        return self.states.get(well_name)

    def set(self, state: WellState) -> None:
        """
        Set the `WellState` for a given well.

        :param state: WellState to store, keyed by `state.well_name`.
        """
        self.states[state.well_name] = state

    @property
    def open_wells(self) -> typing.Iterator[WellState]:
        """Returns Every WellState with is_open=True."""
        return (state for state in self.states.values() if state.is_open)

    @property
    def shut_wells(self) -> typing.Iterator[WellState]:
        """Returns every `WellState` with is_open=False."""
        return (state for state in self.states.values() if not state.is_open)

    def __getitem__(self, well_name: str) -> WellState:
        """
        Get the `WellState` for a given well.

        :raises KeyError: If no state is set for `well_name`.
        """
        state = self.get(well_name)
        if state is None:
            raise KeyError(f"No state set for well {well_name!r}.")
        return state

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.states)

    def __len__(self) -> int:
        return len(self.states)

    def __contains__(self, well_name: typing.Any) -> bool:
        return well_name in self.states

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Returns a  new `WellStates` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `WellStates` with every state converted to target.
        """
        if target == self.unit_system:
            return self
        table = resolve_unit_conversion_table(table)
        return self.__class__(
            states={
                name: state.convert(target, table=table) for name, state in self.states.items()
            },
            unit_system=target,
        )


WellLevel = typing.Literal["well", "perforation"]
"""What a row of `as_dataframe` describes: one well, or one open perforation of a well."""


def get_enum_name(value: object) -> str | None:
    """
    :param value: An enum member, or anything else, or `None`.
    :returns: The member's value as text, or the text of anything else, or `None` for `None`.
    """
    if value is None:
        return None
    return str(value.value) if isinstance(value, enum.Enum) else str(value)


def select_well_names(
    snapshots: typing.Sequence[WellStates], wells: str | typing.Sequence[str] | None
) -> set[str] | None:
    """
    Works out which wells a data frame covers.

    :param snapshots: The well states the data frame is built from.
    :param wells: Well names or patterns such as `"PROD*"`, one or a sequence, or `None` for
        every well.
    :returns: The names of the wells to include, or `None` for every well.
    :raises ValidationError: If a name or pattern matches no well in any state.
    """
    if wells is None:
        return None
    present = {name for snapshot in snapshots for name in snapshot.states}
    chosen: set[str] = set()
    for pattern in [wells] if isinstance(wells, str) else list(wells):
        matches = {name for name in present if fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise ValidationError(
                f"No well matches {pattern!r}. Wells in the states: {sorted(present)}."
            )
        chosen |= matches
    return chosen


def as_dataframe(
    states: WellStates | typing.Iterable[WellStates],
    *,
    backend: Backend = "pandas",
    level: WellLevel = "well",
    times: Number | typing.Sequence[Number] | None = None,
    wells: str | typing.Sequence[str] | None = None,
    open_only: Boolean = False,
    grid: Grid | None = None,
    centroids: Boolean = False,
    unit_system: UnitSystem | None = None,
    categorical: Boolean = False,
) -> AnyDataFrame:
    """
    Turns the well states of one time, or of many times, into a pandas or polars data frame.

    With `level="well"`, the default, there is one row for each well at each time, with its
    control, pressures and rates. With `level="perforation"` there is one row for each open
    perforation of each well at each time, with its cell, flowing pressure and rates. Shut wells
    have no open perforations, so they have no rows at that level.

    Rows of one time are together, and times follow in the order given. A `time` column is the
    first column if `times` is given, or a `snapshot` column (0, 1, 2, ...) when several are
    given without times.

    Well level columns: `well_name`, `well_type` (`"producer"` or `"injector"`), `is_open`,
    `control_mode`, `injected_phase`, `target_rate`, `target_bhp`, `target_thp`, `bhp`, `thp`,
    `oil_rate`, `water_rate`, `gas_rate` (reservoir conditions), `surface_oil_rate`,
    `surface_water_rate`, `surface_gas_rate`, `active_limit` and `active_limit_quantity`. A
    value a well does not have, such as the target it is not controlled by, is missing.

    Perforation level columns: `well_name`, `perforation` (its position among the well's open
    perforations), `cell_index`, `i`, `j` and `k` and optionally `x`, `y` and `z` when a `grid`
    is given, `flowing_pressure`, and `oil_rate`, `water_rate` and `gas_rate`.

    :param states: The well states of one time, or of several times in time order, for example
        the wells of each state of a run.
    :param backend: `"pandas"` or `"polars"`. The library must be installed.
    :param level: `"well"` or `"perforation"`.
    :param times: The elapsed time of each `WellStates`, in the order given, which adds a `time`
        column. A single number is enough for a single `WellStates`.
    :param wells: Well names, or patterns such as `"PROD*"`, one or a sequence, to include. Every
        well if not given.
    :param open_only: For the `"well"` level, leave out shut wells.
    :param grid: The grid the wells are on. For the `"perforation"` level, adds each cell's `i`,
        `j` and `k` when the grid is structured, and is needed by `centroids`.
    :param centroids: For the `"perforation"` level, add the cell centroid coordinates.
    :param unit_system: Unit system to report in. If not given, the states must all share one
        unit system, which is used as it is.
    :param categorical: Store `well_name` and the other text columns as categoricals, which
        saves memory when a long run is reported.
    :returns: A `pandas.DataFrame` or `polars.DataFrame`.
    :raises ValidationError: If an option does not apply to the `level`, a well name matches no
        well, `times` does not match the states, or the states use different unit systems.
    :raises ImportError: If the library behind `backend` is not installed.
    """
    resolve_backend(backend)
    if level not in ("well", "perforation"):
        raise ValidationError(f"`level` must be 'well' or 'perforation'; got {level!r}.")
    if level == "well" and (grid is not None or centroids):
        raise ValidationError("`grid` and `centroids` apply to the 'perforation' level.")
    if level == "perforation" and open_only:
        raise ValidationError(
            "`open_only` applies to the 'well' level. Only open perforations are reported "
            "at the 'perforation' level."
        )

    is_single = isinstance(states, WellStates)
    snapshots = [states] if isinstance(states, WellStates) else list(states)
    for position, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, WellStates):
            raise ValidationError(
                f"`states` must hold `WellStates`; item {position} is a {type(snapshot).__name__}."
            )
    time_values = resolve_times(times, count=len(snapshots))

    if unit_system is None:
        systems = {snapshot.unit_system for snapshot in snapshots}
        if len(systems) > 1:
            raise ValidationError(
                "The states use different unit systems, "
                f"{sorted(system.value for system in systems)}. Pass `unit_system` to report "
                "them all in one."
            )
    else:
        snapshots = [snapshot.convert(unit_system) for snapshot in snapshots]

    names = select_well_names(snapshots, wells)
    rows: list[tuple[int, WellState]] = [
        (position, state)
        for position, snapshot in enumerate(snapshots)
        for state in snapshot.states.values()
        if (names is None or state.well_name in names) and (state.is_open or not open_only)
    ]

    if level == "perforation":
        return build_perforation_dataframe(
            rows,
            backend=backend,
            times=time_values,
            count=len(snapshots),
            is_single=is_single,
            grid=grid,
            centroids=centroids,
            categorical=categorical,
        )
    return build_well_dataframe(
        rows,
        backend=backend,
        times=time_values,
        count=len(snapshots),
        is_single=is_single,
        categorical=categorical,
    )


def count_rows(rows: typing.Sequence[int], *, count: int) -> list[int]:
    """
    :param rows: The snapshot position of each row.
    :param count: How many snapshots there are.
    :returns: How many rows each snapshot has.
    """
    return np.bincount(np.asarray(rows, dtype=np.int64), minlength=count).tolist()


def build_well_dataframe(
    rows: typing.Sequence[tuple[int, WellState]],
    *,
    backend: Backend,
    times: typing.Sequence[Number] | None,
    count: int,
    is_single: bool,
    categorical: bool,
) -> AnyDataFrame:
    """
    :param rows: Each well state to report and the position of the snapshot it came from.
    :param backend: The data frame library.
    :param times: The elapsed time of each snapshot, if given.
    :param count: How many snapshots there are.
    :param is_single: Whether a single `WellStates` was given.
    :param categorical: Whether to store text columns as categoricals.
    :returns: One row for each well state.
    """
    states = [state for _, state in rows]

    def numbers(values: typing.Iterable[Number | None]) -> np.ndarray:
        return np.fromiter((np.nan if v is None else v for v in values), dtype=np.float64)

    def text(values: typing.Iterable[str | None], name: str) -> Column:
        return Column(name=name, values=list(values), kind="str", categorical=categorical)

    def target(state: WellState, name: str) -> Number | None:
        return getattr(state.active_control, name, None)

    def well_type(state: WellState) -> str | None:
        if isinstance(state.active_control, ProducerControl):
            return "producer"
        if isinstance(state.active_control, InjectorControl):
            return "injector"
        return None

    columns: list[Column] = []
    label = get_label_column(
        times=times, row_counts=count_rows([p for p, _ in rows], count=count), is_single=is_single
    )
    if label is not None:
        columns.append(label)
    columns += [
        text((s.well_name for s in states), "well_name"),
        text((well_type(s) for s in states), "well_type"),
        Column(name="is_open", values=[s.is_open for s in states], kind="bool"),
        text(
            (get_enum_name(getattr(s.active_control, "mode", None)) for s in states),
            "control_mode",
        ),
        text(
            (get_enum_name(getattr(s.active_control, "injected_phase", None)) for s in states),
            "injected_phase",
        ),
        Column(
            name="target_rate",
            values=numbers(target(s, "target_rate") for s in states),
            missing_is_null=True,
        ),
        Column(
            name="target_bhp",
            values=numbers(target(s, "target_bhp") for s in states),
            missing_is_null=True,
        ),
        Column(
            name="target_thp",
            values=numbers(target(s, "target_thp") for s in states),
            missing_is_null=True,
        ),
        Column(name="bhp", values=numbers(s.bhp for s in states)),
        Column(name="thp", values=numbers(s.thp for s in states), missing_is_null=True),
        Column(name="oil_rate", values=numbers(s.rates.oil for s in states)),
        Column(name="water_rate", values=numbers(s.rates.water for s in states)),
        Column(name="gas_rate", values=numbers(s.rates.gas for s in states)),
        Column(name="surface_oil_rate", values=numbers(s.surface_rates.oil for s in states)),
        Column(name="surface_water_rate", values=numbers(s.surface_rates.water for s in states)),
        Column(name="surface_gas_rate", values=numbers(s.surface_rates.gas for s in states)),
        text(
            (
                getattr(s.active_limit, "__type__", type(s.active_limit).__name__)
                if s.active_limit is not None
                else None
                for s in states
            ),
            "active_limit",
        ),
        text(
            (get_enum_name(getattr(s.active_limit, "quantity", None)) for s in states),
            "active_limit_quantity",
        ),
    ]
    return build_dataframe(columns, backend=backend)


def build_perforation_dataframe(
    rows: typing.Sequence[tuple[int, WellState]],
    *,
    backend: Backend,
    times: typing.Sequence[Number] | None,
    count: int,
    is_single: bool,
    grid: Grid | None,
    centroids: bool,
    categorical: bool,
) -> AnyDataFrame:
    """
    :param rows: Each well state to report and the position of the snapshot it came from.
    :param backend: The data frame library.
    :param times: The elapsed time of each snapshot, if given.
    :param count: How many snapshots there are.
    :param is_single: Whether a single `WellStates` was given.
    :param grid: The grid the wells are on, if given.
    :param centroids: Whether to add cell centroid coordinates.
    :param categorical: Whether to store text columns as categoricals.
    :returns: One row for each open perforation of each well state.
    """
    snapshot_positions: list[int] = []
    well_names: list[str] = []
    ordinals: list[int] = []
    perforations: list[PerforationState] = []
    for position, state in rows:
        for ordinal, perforation in enumerate(state.perforation_states):
            snapshot_positions.append(position)
            well_names.append(state.well_name)
            ordinals.append(ordinal)
            perforations.append(perforation)

    cell_indices = np.fromiter((p.cell_index for p in perforations), dtype=np.int64)
    columns: list[Column] = []
    label = get_label_column(
        times=times, row_counts=count_rows(snapshot_positions, count=count), is_single=is_single
    )
    if label is not None:
        columns.append(label)
    columns += [
        Column(name="well_name", values=well_names, kind="str", categorical=categorical),
        Column(name="perforation", values=ordinals, kind="int"),
        Column(name="cell_index", values=cell_indices, kind="int"),
        *get_grid_columns(grid=grid, cell_indices=cell_indices, centroids=centroids),
        Column(
            name="flowing_pressure",
            values=np.fromiter((p.flowing_pressure for p in perforations), dtype=np.float64),
        ),
        Column(
            name="oil_rate",
            values=np.fromiter((p.rates.oil for p in perforations), dtype=np.float64),
        ),
        Column(
            name="water_rate",
            values=np.fromiter((p.rates.water for p in perforations), dtype=np.float64),
        ),
        Column(
            name="gas_rate",
            values=np.fromiter((p.rates.gas for p in perforations), dtype=np.float64),
        ),
    ]
    return build_dataframe(columns, backend=backend)
