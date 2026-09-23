"""Per-timestep dynamic well state."""

import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.errors import ValidationError
from bores.serde.stores import StoreSerializable
from bores.types import Number, UnitConversionTable, UnitSystem
from bores.utils import scale
from bores.wells.base import Perforation
from bores.wells.controls import Limit, WellControl

__all__ = [
    "ConnectionSample",
    "PerforationState",
    "PhaseValues",
    "WellState",
    "WellStates",
]


class PhaseValues(typing.NamedTuple):
    """A value for each of the three fluid phases."""

    oil: Number
    """The value for the oil phase."""

    water: Number
    """The value for the water phase."""

    gas: Number
    """The value for the gas phase."""


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


@attrs.frozen(kw_only=True, slots=True)
class PerforationState(StoreSerializable):
    """Per-perforation dynamic snapshot for one timestep."""

    perforation: Perforation
    """The perforation this snapshot is for."""

    cell_index: int
    """The grid cell this perforation connects to."""

    flowing_pressure: Number
    """Flowing bottomhole pressure at this perforation."""

    phase_rates: PhaseValues
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
            phase_rates=PhaseValues(
                oil=scale(self.phase_rates.oil, rate_factor),
                water=scale(self.phase_rates.water, rate_factor),
                gas=scale(self.phase_rates.gas, rate_factor),
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

    phase_rates: PhaseValues
    """Rate of each phase for the whole well, at reservoir conditions."""

    surface_phase_rates: PhaseValues
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
        return self.phase_rates.oil + self.phase_rates.water

    @property
    def total_surface_liquid_rate(self) -> Number:
        """Combined oil and water rate for the well, at surface conditions."""
        return self.surface_phase_rates.oil + self.surface_phase_rates.water

    def get_perforation_state(self, cell_index: int) -> PerforationState:
        """
        Look up and return the `PerforationState` connected to a given cell.

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
        :returns: This snapshot, with `bhp`, `thp`, `phase_rates`,
            `surface_phase_rates`, `perforation_states`, `active_control`,
            and `active_limit` (if set) converted to `target`.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        reservoir_rate_factor = factors["reservoir_rate"]
        return attrs.evolve(
            self,
            bhp=scale(self.bhp, factors["pressure"]),
            thp=scale(self.thp, factors["pressure"]) if self.thp is not None else None,
            phase_rates=PhaseValues(
                oil=scale(self.phase_rates.oil, reservoir_rate_factor),
                water=scale(self.phase_rates.water, reservoir_rate_factor),
                gas=scale(self.phase_rates.gas, reservoir_rate_factor),
            ),
            surface_phase_rates=PhaseValues(
                oil=scale(self.surface_phase_rates.oil, factors["liquid_surface_rate"]),
                water=scale(self.surface_phase_rates.water, factors["liquid_surface_rate"]),
                gas=scale(self.surface_phase_rates.gas, factors["gas_surface_rate"]),
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
        "unit_system": typing.Optional[UnitSystem],  # noqa: UP045
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

    def __contains__(self, well_name: object) -> bool:
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
        return self.__class__(
            states={
                name: state.convert(target, table=table) for name, state in self.states.items()
            },
            unit_system=target,
        )
