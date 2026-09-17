"""Convenience factories for `ScheduleItem`s/`Schedule`s."""

from bores.schedule.base import Action, Event, ModelT, Schedule, ScheduleItem, SerializableEvent
from bores.schedule.events import AllOf, AnyOf, TimeEvent

__all__ = ["all_of", "any_of", "at", "item", "schedule"]


def at(*, time: float, action: Action[ModelT], name: str | None = None) -> ScheduleItem[ModelT]:
    """
    Builds a `Rule` that runs `action` once, at `time`.

    :param time: Elapsed time to fire at.
    :param action: The action to run.
    :param name: Optional label for the rule.
    :returns: A `Rule` pairing a `TimeEvent` with `action`.
    """
    return ScheduleItem(
        event=TimeEvent(at=time), action=action, name=name or f"at({time!r}, {action!r})"
    )


def item(
    *, event: Event[ModelT], action: Action[ModelT], name: str | None = None
) -> "ScheduleItem[ModelT]":
    """
    Builds a `ScheduleItem` from an event and an action.

    :param event: Fires (or not) to decide whether `action` runs.
    :param action: Runs when `event` fires.
    :param name: Optional label for the item.
    :returns: The `Rule`.
    """
    return ScheduleItem(event=event, action=action, name=name or f"item({event!r}, {action!r})")


def schedule(*rules: ScheduleItem[ModelT]) -> Schedule[ModelT]:
    """
    Builds a `Schedule` from a set of rules.

    :param rules: Every rule for the schedule, in firing-check order.
    :returns: The `Schedule`.
    """
    return Schedule(items=tuple(rules))


def all_of(*events: SerializableEvent) -> AllOf:
    """
    Builds an `AllOf` from a set of events.

    :param events: Every event that must fire.
    :returns: The `AllOf`.
    """
    return AllOf(events=tuple(events))


def any_of(*events: SerializableEvent) -> AnyOf:
    """
    Builds an `AnyOf` from a set of events.

    :param events: Every event checked; any one firing is enough.
    :returns: The `AnyOf`.
    """
    return AnyOf(events=tuple(events))
