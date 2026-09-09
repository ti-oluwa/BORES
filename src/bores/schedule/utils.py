"""Small convenience builders on top of `bores.schedule.base`/`events`."""

from bores.schedule.base import Action, Event, ModelT, Rule, Schedule, SerializableEvent
from bores.schedule.events import AllOf, AnyOf

__all__ = ["all_of", "any_of", "at", "rule", "schedule"]


def at(time: float, action: Action[ModelT], *, name: str | None = None) -> Rule[ModelT]:
    """
    Shorthand for the common case: run `action` once, at `time`.

    :param time: Elapsed time to fire at.
    :param action: The action to run.
    :param name: Optional label for the rule.
    :returns: A `Rule` pairing a `TimeEvent` with `action`.
    """
    from bores.schedule.events import TimeEvent

    return Rule(event=TimeEvent(at=time), action=action, name=name)


def rule(event: Event[ModelT], action: Action[ModelT], *, name: str | None = None) -> Rule[ModelT]:
    """
    Shorthand for `Rule(event=..., action=..., name=...)`.

    :param event: Fires (or not) to decide whether `action` runs.
    :param action: Runs when `event` fires.
    :param name: Optional label for the rule.
    :returns: The `Rule`.
    """
    return Rule(event=event, action=action, name=name)


def schedule(*rules: Rule[ModelT]) -> Schedule[ModelT]:
    """
    Shorthand for `Schedule(rules=rules)`.

    :param rules: Every rule for the schedule, in firing-check order.
    :returns: The `Schedule`.
    """
    return Schedule(rules=tuple(rules))


def all_of(*events: SerializableEvent) -> AllOf:
    """Shorthand for `AllOf(events=events)`."""
    return AllOf(events=tuple(events))


def any_of(*events: SerializableEvent) -> AnyOf:
    """Shorthand for `AnyOf(events=events)`."""
    return AnyOf(events=tuple(events))
