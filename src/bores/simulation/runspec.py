"""Run-level configuration, passed through `ScheduleContext.runspec`."""

import datetime

import attrs

from bores.constants import Constants
from bores.serde.stores import StoreSerializable
from bores.simulation.timing import Timer
from bores.types import UnitSystem

__all__ = ["RunSpec"]


@attrs.frozen(kw_only=True, slots=True)
class RunSpec(
    StoreSerializable,
    fields={
        "unit_system": UnitSystem,
        "constants": Constants,
        "timer": Timer | None,
        "output_frequency": int,
        "start_date": datetime.datetime | None,
    },
):
    """Run-level configuration for a simulation."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for the run."""

    constants: Constants = attrs.field(factory=Constants)
    """Physical and conversion constants for the run."""

    timer: Timer | None = None
    """The run's time manager, if one has been attached yet."""

    output_frequency: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    """Frequency, in time steps, at which model states are output."""

    start_date: datetime.datetime | None = None
    """The run's calendar start date, if the run is calendar-anchored."""
