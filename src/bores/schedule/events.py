"""Builtin `Event` implementations: `TimeEvent`, `IntervalEvent`, `ThresholdEvent`, `AllOf`, `AnyOf`."""

import enum
import operator
import typing

import attrs

from bores.errors import ValidationError
from bores.schedule.base import (
    ModelT,
    ScheduleContext,
    SerializableEvent,
    event_type,
)
from bores.types import Boolean, Number

__all__ = [
    "COMPARISON_FUNCTIONS",
    "AllOf",
    "AnyOf",
    "ComparisonOperator",
    "IntervalEvent",
    "ThresholdEvent",
    "TimeEvent",
    "TimeStepEvent",
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


@event_type
@attrs.frozen(kw_only=True, slots=True)
class TimeEvent(SerializableEvent[ModelT]):
    """Fires once, the first time the schedule is advanced past `at`."""

    at: Number
    """Elapsed time this event fires at, in the schedule's unit system."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when `context.time` crosses `at` since `context.previous_time`.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context.
        :returns: Whether `at` was just crossed.
        """
        return context.previous_time < self.at <= context.time


@event_type
@attrs.frozen(kw_only=True, slots=True)
class TimeStepEvent(SerializableEvent[ModelT]):
    """Fires once, the first time the schedule is advanced past step `at`."""

    at: int
    """The time-step index this event fires at."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when `context.time_step` crosses `at` since `context.previous_time_step`.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context. `context.time_step`
            and `context.previous_time_step` must both be set.
        :returns: Whether `at` was just crossed.
        """
        if context.time_step is None or context.previous_time_step is None:
            raise ValidationError(
                f"{type(self).__name__} needs `context.time_step` and "
                "`context.previous_time_step` to both be set."
            )
        return context.previous_time_step < self.at <= context.time_step


@event_type
@attrs.frozen(kw_only=True, slots=True)
class IntervalEvent(SerializableEvent[ModelT]):
    """Fires every `every` time units, starting at `start`."""

    every: Number
    """How often this event fires."""

    start: Number = 0.0
    """The first time this event is eligible to fire."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when an interval boundary was crossed since `context.previous_time`.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context.
        :returns: Whether an interval boundary was just crossed.
        """
        if context.time < self.start:
            return False
        previous_count = self.get_boundary_count(time=context.previous_time)
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
        return int((time - self.start) // self.every) + 1


@attrs.frozen(kw_only=True, slots=True)
class ThresholdEvent(SerializableEvent[ModelT]):
    """Fires when `get_value` crosses `threshold`, per `op`. Subclass and implement `get_value`."""

    __abstract_serializable__ = True

    op: ComparisonOperator
    """How `get_value(...)` is compared against `threshold`."""

    threshold: Number
    """The value `get_value(...)` is compared against."""

    def get_value(self, *, model: ModelT, context: ScheduleContext) -> Number:
        """
        Reads the value this event watches. Must be overridden.

        :param model: The model being scheduled against.
        :param context: The current moment's context. `context.state`
            carries the latest solved values, if `get_value` needs one
            that isn't part of the static `model` itself.
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


@event_type
@attrs.frozen(kw_only=True, slots=True)
class AllOf(SerializableEvent[ModelT]):
    """Fires only when every one of `events` fires."""

    events: tuple[SerializableEvent, ...] = attrs.field(converter=tuple)
    """Every event that must fire for this one to fire."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when every one of `events` fires.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: Whether every sub-event fired.
        """
        return all(event(model, context) for event in self.events)


@event_type
@attrs.frozen(kw_only=True, slots=True)
class AnyOf(SerializableEvent[ModelT]):
    """Fires when any one of `events` fires."""

    events: tuple[SerializableEvent, ...] = attrs.field(converter=tuple)
    """Every event checked; any one firing is enough."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Fires when any one of `events` fires.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: Whether any sub-event fired.
        """
        return any(event(model, context) for event in self.events)
