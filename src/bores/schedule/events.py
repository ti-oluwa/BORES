"""Builtin `Event` implementations: `TimeEvent`, `IntervalEvent`, `ThresholdEvent`, `AllOf`, `AnyOf`."""

import enum
import operator
import typing

import attrs

from bores.schedule.base import ModelT, ScheduleContext, SerializableEvent, event_type
from bores.types import Boolean, Number

__all__ = [
    "AllOf",
    "AnyOf",
    "ComparisonOperator",
    "IntervalEvent",
    "ThresholdEvent",
    "TimeEvent",
]


class ComparisonOperator(str, enum.Enum):
    """Deck-agnostic comparison, for `ThresholdEvent` and anything like it."""

    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    EQ = "eq"


COMPARISON_FUNCTIONS = {
    ComparisonOperator.GT: operator.gt,
    ComparisonOperator.GE: operator.ge,
    ComparisonOperator.LT: operator.lt,
    ComparisonOperator.LE: operator.le,
    ComparisonOperator.EQ: operator.eq,
}


@event_type
@attrs.frozen(kw_only=True, slots=True)
class TimeEvent(SerializableEvent[ModelT]):
    """Fires once, the first time the schedule is advanced past `at`."""

    at: Number
    """Elapsed time this event fires at, in the schedule's unit system."""

    def __call__(self, model: typing.Any, context: ScheduleContext) -> Boolean:
        return context.previous_time < self.at <= context.time


@event_type
@attrs.frozen(kw_only=True, slots=True)
class IntervalEvent(SerializableEvent[ModelT]):
    """Fires every `every` time units, starting at `start`."""

    every: Number
    """How often this event fires."""

    start: Number = 0.0
    """The first time this event is eligible to fire."""

    def __call__(self, model: typing.Any, context: ScheduleContext) -> Boolean:
        if context.time < self.start:
            return False
        # Number of interval boundaries at/before previous_time vs. time -
        # if that count changed, an interval boundary was just crossed.
        previous_count = self._boundary_count(context.previous_time)
        current_count = self._boundary_count(context.time)
        return current_count > previous_count

    def _boundary_count(self, time: Number) -> int:
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

    def get_value(self, model: ModelT, context: ScheduleContext) -> Number:
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
        value = self.get_value(model, context)
        return COMPARISON_FUNCTIONS[self.op](value, self.threshold)


@event_type
@attrs.frozen(kw_only=True, slots=True)
class AllOf(SerializableEvent[ModelT]):
    """Fires only when every one of `events` fires."""

    events: tuple[SerializableEvent, ...] = attrs.field(converter=tuple)

    def __call__(self, model: typing.Any, context: ScheduleContext) -> Boolean:
        return all(event(model, context) for event in self.events)


@event_type
@attrs.frozen(kw_only=True, slots=True)
class AnyOf(SerializableEvent[ModelT]):
    """Fires when any one of `events` fires."""

    events: tuple[SerializableEvent, ...] = attrs.field(converter=tuple)

    def __call__(self, model: typing.Any, context: ScheduleContext) -> Boolean:
        return any(event(model, context) for event in self.events)
