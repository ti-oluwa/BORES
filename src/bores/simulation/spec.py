"""Run-level configuration, passed through `ScheduleContext.runspec`."""

import datetime

import attrs

from bores.constants import Constants
from bores.serde.stores import StoreSerializable
from bores.simulation.timing import Timer
from bores.types import UnitSystem
from bores.wells.resolution.spec import WellControlSpec

__all__ = ["RunSpec"]


@attrs.frozen(kw_only=True, slots=True)
class RunSpec(StoreSerializable):
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

    control_spec: WellControlSpec = attrs.field(factory=WellControlSpec)
    """Numerical tuning for well-control resolution. Defaults to `WellControlSpec`'s own defaults."""
