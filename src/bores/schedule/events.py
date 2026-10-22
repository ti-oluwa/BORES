"""Builtin `Event` implementations."""

import enum
import math
import operator
import typing
from datetime import datetime

import attrs

from bores.errors import ValidationError
from bores.schedule.base import Event, ModelT, ScheduleContext
from bores.timestep.targets import PeriodicTarget, TimeTargets
from bores.types import Boolean, Number, UnitSystem
from bores.utils import get_current_date, get_current_time

__all__ = [
    "COMPARISON_FUNCTIONS",
    "AllOf",
    "AnyOf",
    "ComparisonOperator",
    "DateEvent",
    "IntervalEvent",
    "ThresholdEvent",
    "TimeEvent",
    "TimeStepEvent",
    "TimeWindowEvent",
]


class ComparisonOperator(str, enum.Enum):
    """Deck-agnostic comparison, for `ThresholdEvent` and anything like it."""

    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    EQ = "eq"


COMPARISON_FUNCTIONS: dict[ComparisonOperator, typing.Callable[[Number, Number], bool]] = {
    ComparisonOperator.GT: operator.gt,
    ComparisonOperator.GE: operator.ge,
    ComparisonOperator.LT: operator.lt,
    ComparisonOperator.LE: operator.le,
    ComparisonOperator.EQ: operator.eq,
}
"""Maps each `ComparisonOperator` to the function it applies."""


@attrs.frozen(kw_only=True, slots=True)
class TimeEvent(Event[ModelT]):
    """Fires once, the first time the schedule is advanced past `at`."""

    at: Number
    """Elapsed time this event fires at, in the schedule's unit system."""

    def get_time_targets(
        self, *, start_date: datetime | None = None, unit_system: UnitSystem = UnitSystem.FIELD
    ) -> TimeTargets:
        """
        Return the time at which this event fires.

        :param start_date: The simulation's start date. Unused.
        :param unit_system: Unit system of the run's time. Unused, `at` is already in it.
        :returns: The time this event fires at, so a time stepper lands on it exactly.
        """
        return TimeTargets(times=[self.at])

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when `context.time` crosses `at` since `context.previous_time`.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context.
        :returns: Whether `at` was just crossed.
        """
        previous_time = context.previous_time
        return (previous_time is None or previous_time < self.at) and self.at <= context.time


@attrs.frozen(kw_only=True, slots=True)
class TimeWindowEvent(Event[ModelT]):
    """
    Fires on every advance while the schedule's time lies in `[start, end)`.

    Unlike `TimeEvent` this stays true for the whole window, so combined with a condition
    through `AllOf` it makes that condition live only between two moments.
    """

    start: Number
    """Elapsed time the window opens at, in the schedule's unit system."""

    end: Number | None = None
    """Elapsed time the window closes at. `None` keeps it open for the rest of the run."""

    def get_time_targets(
        self, *, start_date: datetime | None = None, unit_system: UnitSystem = UnitSystem.FIELD
    ) -> TimeTargets:
        """
        Return the times at which this window opens and closes.

        :param start_date: The simulation's start date. Unused.
        :param unit_system: Unit system of the run's time. Unused, `start` and `end` are already in it.
        :returns: The times the window opens and closes at, so a time stepper lands on both.
        """
        times = [self.start] if self.end is None else [self.start, self.end]
        return TimeTargets(times=times)

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires while `context.time` is inside the window.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context.
        :returns: Whether `start <= context.time < end`.
        """
        return self.start <= context.time and (self.end is None or context.time < self.end)


@attrs.frozen(kw_only=True, slots=True)
class TimeStepEvent(Event[ModelT]):
    """Fires once, the first time the schedule is advanced past step `at`."""

    at: int
    """The time-step index this event fires at."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when `context.time_step` crosses `at` since `context.previous_time_step`.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context. `context.time_step`
            must be set. A `None` `context.previous_time_step` means no step
            has been advanced to yet.
        :returns: Whether `at` was just crossed.
        """
        if context.time_step is None:
            raise ValidationError(f"{type(self).__name__} needs `context.time_step` to be set.")
        previous_step = context.previous_time_step
        return (previous_step is None or previous_step < self.at) and self.at <= context.time_step


@attrs.frozen(kw_only=True, slots=True)
class DateEvent(Event[ModelT]):
    """Fires once, the first time the schedule is advanced past `at`."""

    at: datetime = attrs.field(
        converter=lambda value: datetime.fromisoformat(value) if isinstance(value, str) else value
    )
    """The date this event fires at, given in ISO 8601 format or as a `datetime.datetime`."""

    def get_time_targets(
        self, *, start_date: datetime | None = None, unit_system: UnitSystem = UnitSystem.FIELD
    ) -> TimeTargets:
        """
        Return the elapsed time at which this date event fires.

        :param start_date: The simulation's start date. Required.
        :param unit_system: Unit system the elapsed time is expressed in.
        :returns: The elapsed time of `at`, so a time stepper lands on it exactly.
        :raises ValidationError: If `start_date` is not given.
        """
        if start_date is None:
            raise ValidationError(f"{type(self).__name__} needs `start_date` to find its time.")
        elapsed = get_current_time(
            start_date=start_date, current_date=self.at, unit_system=unit_system
        )
        return TimeTargets(times=[elapsed])

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when `context.start_date + context.time` crosses `at` since
        `context.start_date + context.previous_time`.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context. `context.start_date`
            must be set.
        :returns: Whether `at` was just crossed.
        """
        if context.start_date is None:
            raise ValidationError(f"{type(self).__name__} needs `context.start_date` to be set.")

        current_date = get_current_date(
            start_date=context.start_date,
            elapsed_time=context.time,
            unit_system=context.unit_system,
        )
        previous_date = (
            get_current_date(
                start_date=context.start_date,
                elapsed_time=context.previous_time,
                unit_system=context.unit_system,
            )
            if context.previous_time is not None
            else None
        )
        return (previous_date is None or previous_date < self.at) and self.at <= current_date


@attrs.frozen(kw_only=True, slots=True)
class IntervalEvent(Event[ModelT]):
    """Fires every `every` time units, starting at `start`."""

    every: Number = attrs.field(validator=attrs.validators.gt(0))
    """How often this event fires. Must be positive."""

    start: Number = 0.0
    """The first time this event is eligible to fire."""

    def get_time_targets(
        self, *, start_date: datetime | None = None, unit_system: UnitSystem = UnitSystem.FIELD
    ) -> TimeTargets:
        """
        Return every recurring time at which this event fires.

        :param start_date: The simulation's start date. Unused.
        :param unit_system: Unit system of the run's time. Unused, `every` and `start` are already in it.
        :returns: Every boundary this event fires at, so a time stepper lands on each.
        """
        return TimeTargets(periodic=[PeriodicTarget(every=self.every, start=self.start)])

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when an interval boundary was crossed since `context.previous_time`.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context.
        :returns: Whether an interval boundary was just crossed.
        """
        if context.time < self.start:
            return False
        previous_count = (
            0
            if context.previous_time is None
            else self.get_boundary_count(time=context.previous_time)
        )
        current_count = self.get_boundary_count(time=context.time)
        return current_count > previous_count

    def get_boundary_count(self, *, time: Number) -> int:
        """
        Counts how many interval boundaries fall at or before `time`.

        :param time: Elapsed time to count boundaries up to.
        :returns: The number of boundaries at or before `time`.
        """
        if time < self.start:
            return 0
        # The tiny tolerance keeps a boundary from being missed when an accumulated
        # floating-point time lands just below it, e.g. `0.1 + 0.1 + 0.1 + 0.1 + 0.1`.
        return math.floor((time - self.start) / self.every + 1e-9) + 1


@attrs.frozen(kw_only=True, slots=True)
class ThresholdEvent(Event[ModelT]):
    """
    Fires whenever `get_value` satisfies `op` against `threshold`. Subclass and implement `get_value`.

    The event fires at every check while the comparison holds, not only when it first becomes
    true. A `ScheduleItem` built on it therefore runs its action once unless told otherwise.
    """

    level_triggered: typing.ClassVar[bool] = True
    """Marks an event that holds across many checks, so its item defaults to a single run."""

    op: ComparisonOperator
    """How `get_value(...)` is compared against `threshold`."""

    threshold: Number
    """The value `get_value(...)` is compared against."""

    def get_value(self, *, model: ModelT, context: ScheduleContext) -> Number:
        """
        Reads the value this event watches. Must be overridden.

        :param model: The model being scheduled against.
        :param context: The current moment's context. `context.extra` carries
            anything the caller supplied that is not part of `model` itself.
        :returns: The current value of whatever this event watches.
        """
        raise NotImplementedError

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when `get_value(model, context)` satisfies `op` against `threshold`.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: Whether the comparison holds.
        """
        value = self.get_value(model=model, context=context)
        return COMPARISON_FUNCTIONS[self.op](value, self.threshold)


@attrs.frozen(kw_only=True, slots=True)
class AllOf(Event[ModelT]):
    """Fires only when every one of `events` fires."""

    events: tuple[Event[ModelT], ...] = attrs.field(converter=tuple)
    """Every event that must fire for this one to fire."""

    @property
    def level_triggered(self) -> bool:
        """Whether any sub-event holds across many checks."""
        return any(getattr(event, "level_triggered", False) for event in self.events)

    def get_time_targets(
        self, *, start_date: datetime | None = None, unit_system: UnitSystem = UnitSystem.FIELD
    ) -> TimeTargets:
        """
        Return the combined time targets of all sub-events.

        :param start_date: The simulation's start date.
        :param unit_system: Unit system of the run's time.
        :returns: Every time any of the events needs landed on. Landing on more times than the combined event fires at is harmless.
        """
        times: list[Number] = []
        periodic: list[PeriodicTarget] = []
        for event in self.events:
            getter = getattr(event, "get_time_targets", None)
            if getter is None:
                continue

            targets = getter(start_date=start_date, unit_system=unit_system)
            times.extend(targets.times)
            periodic.extend(targets.periodic)
        return TimeTargets(times=times, periodic=periodic)

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when every one of `events` fires.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: Whether every sub-event fired.
        """
        return all(event(model, context) for event in self.events)


@attrs.frozen(kw_only=True, slots=True)
class AnyOf(Event[ModelT]):
    """Fires when any one of `events` fires."""

    events: tuple[Event[ModelT], ...] = attrs.field(converter=tuple)
    """Every event checked; any one firing is enough."""

    @property
    def level_triggered(self) -> bool:
        """Whether any sub-event holds across many checks."""
        return any(getattr(event, "level_triggered", False) for event in self.events)

    def get_time_targets(
        self, *, start_date: datetime | None = None, unit_system: UnitSystem = UnitSystem.FIELD
    ) -> TimeTargets:
        """
        Return the combined time targets of all sub-events.

        :param start_date: The simulation's start date.
        :param unit_system: Unit system of the run's time.
        :returns: Every time any of the events needs landed on.
        """
        times: list[Number] = []
        periodic: list[PeriodicTarget] = []
        for event in self.events:
            getter = getattr(event, "get_time_targets", None)
            if getter is None:
                continue

            targets = getter(start_date=start_date, unit_system=unit_system)
            times.extend(targets.times)
            periodic.extend(targets.periodic)
        return TimeTargets(times=times, periodic=periodic)

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when any one of `events` fires.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: Whether any sub-event fired.
        """
        return any(event(model, context) for event in self.events)
