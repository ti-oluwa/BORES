"""Convenience builders for `Rule`/`Schedule` construction: `at`, `rule`, `schedule`, `all_of`, `any_of`."""

from bores.schedule.base import Action, Event, ModelT, Rule, Schedule, SerializableEvent
from bores.schedule.events import AllOf, AnyOf, TimeEvent

__all__ = ["all_of", "any_of", "at", "rule", "schedule"]


def at(*, time: float, action: Action[ModelT], name: str | None = None) -> Rule[ModelT]:
    """
    Builds a `Rule` that runs `action` once, at `time`.

    :param time: Elapsed time to fire at.
    :param action: The action to run.
    :param name: Optional label for the rule.
    :returns: A `Rule` pairing a `TimeEvent` with `action`.
    """
    return Rule(event=TimeEvent(at=time), action=action, name=name)


def rule(
    *, event: Event[ModelT], action: Action[ModelT], name: str | None = None
) -> "Rule[ModelT]":
    """
    Builds a `Rule` from an event and an action.

    :param event: Fires (or not) to decide whether `action` runs.
    :param action: Runs when `event` fires.
    :param name: Optional label for the rule.
    :returns: The `Rule`.
    """
    return Rule(event=event, action=action, name=name)


def schedule(*rules: Rule[ModelT]) -> Schedule[ModelT]:
    """
    Builds a `Schedule` from a set of rules.

    :param rules: Every rule for the schedule, in firing-check order.
    :returns: The `Schedule`.
    """
    return Schedule(rules=tuple(rules))


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
