"""Generic event-driven scheduling: `Event`/`Action` protocols, `Rule`, `Schedule`."""

import threading
import typing

import attrs
from typing_extensions import Self

from bores.errors import ActionError, EventError, StopSimulation
from bores.serde.base import Serializable
from bores.serde.registry import make_serializable_type_registrar
from bores.simulation.runspec import RunSpec
from bores.types import Boolean, Number, UnitSystem

__all__ = [
    "ACTION_TYPES",
    "EVENT_TYPES",
    "PASSTHROUGH_EXCEPTIONS",
    "Action",
    "Event",
    "ModelT",
    "Rule",
    "Schedule",
    "ScheduleContext",
    "SerializableAction",
    "SerializableEvent",
    "action_type",
    "event_type",
]

ModelT = typing.TypeVar("ModelT")
ModelTcon = typing.TypeVar("ModelTcon", contravariant=True)

PASSTHROUGH_EXCEPTIONS: tuple[type[BaseException], ...] = (StopSimulation,)
"""Exceptions `Schedule.apply` never wraps in `EventError`/`ActionError` - re-raised as is."""


@attrs.frozen(kw_only=True, slots=True)
class ScheduleContext:
    """Context passed to every `Event`/`Action` call alongside the model."""

    time: Number
    """Elapsed time the schedule is being advanced to, in `unit_system`."""

    previous_time: Number = 0.0
    """Elapsed time the schedule was last advanced to."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `time`, `previous_time`, and any value an event reads."""

    runspec: RunSpec | None = None
    """The run's configuration, if any."""

    state: object | None = None
    """The latest solved state, if any (a `CompiledWellResolution`, for example)."""

    time_step: int | None = None
    """The current time-step index, if the caller is tracking one."""

    previous_time_step: int | None = None
    """The time-step index the schedule was last advanced at. `TimeStepEvent` needs this to
    tell "just reached this step" apart from "still past it next call too"."""

    step_size: Number | None = None
    """The current time-step's size, in `unit_system`, if the caller is tracking one."""

    extra: typing.Mapping[str, object] = attrs.field(factory=dict)
    """Additional domain-specific context."""


@typing.runtime_checkable
class Event(typing.Protocol[ModelTcon]):
    """A trigger. Any callable matching this signature satisfies it."""

    def __call__(self, model: ModelTcon, context: ScheduleContext) -> Boolean:
        """
        Evaluates whether the paired action should fire.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: Whether the paired action should fire.
        """
        ...


@typing.runtime_checkable
class Action(typing.Protocol[ModelT]):
    """A model change. Any callable matching this signature satisfies it."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Applies this action to `model`.

        :param model: The model to change.
        :param context: The current moment's context.
        :returns: The changed model.
        """
        ...


EVENT_TYPES: dict[str, type["SerializableEvent"]] = {}
ACTION_TYPES: dict[str, type["SerializableAction"]] = {}


class SerializableEvent(Serializable, typing.Generic[ModelT]):
    """Base for `Event` implementations that support `dump`/`load`."""

    __abstract_serializable__ = True

    def __call__(self, model: ModelT, context: ScheduleContext) -> Boolean:
        """
        Evaluates whether the paired action should fire. Must be overridden.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: Whether the paired action should fire.
        """
        raise NotImplementedError


class SerializableAction(Serializable, typing.Generic[ModelT]):
    """Base for `Action` implementations that support `dump`/`load`."""

    __abstract_serializable__ = True

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Applies this action to `model`. Must be overridden.

        :param model: The model to change.
        :param context: The current moment's context.
        :returns: The changed model.
        """
        raise NotImplementedError


event_type = make_serializable_type_registrar(
    base_cls=SerializableEvent,
    registry=EVENT_TYPES,
    lock=threading.Lock(),
    key_attr="__type__",
)
action_type = make_serializable_type_registrar(
    base_cls=SerializableAction,
    registry=ACTION_TYPES,
    lock=threading.Lock(),
    key_attr="__type__",
)


class Rule(
    typing.Generic[ModelT],
    Serializable,
    fields={"event": SerializableEvent, "action": SerializableAction, "name": str | None},
):
    """One `(Event, Action)` pairing: `action` fires whenever `event` fires."""

    def __init__(
        self,
        *,
        event: Event[ModelT],
        action: Action[ModelT],
        name: str | None = None,
    ) -> None:
        """
        :param event: Decides whether `action` runs.
        :param action: Changes the model when `event` fires.
        :param name: Optional label.
        """
        self.event = event
        self.action = action
        self.name = name

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"{type(self).__name__}({self.event!r}, {self.action!r}{label})"


@attrs.frozen(kw_only=True, slots=True)
class Schedule(typing.Generic[ModelT]):
    """Every rule for a run. Advances a model by applying whichever rules fire."""

    rules: tuple[Rule[ModelT], ...] = attrs.field(converter=tuple)

    def apply(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Applies every rule whose event fires, in `rules` order.

        :param model: The model to advance.
        :param context: The current moment's context.
        :returns: The model after every firing rule's action has run.
        :raises EventError: If a rule's event raises anything other than
            a `PASSTHROUGH_EXCEPTIONS` member.
        :raises ActionError: If a rule's action raises anything other
            than a `PASSTHROUGH_EXCEPTIONS` member.
        """
        for rule in self.rules:
            try:
                fires = rule.event(model, context)
            except PASSTHROUGH_EXCEPTIONS:
                raise
            except Exception as exc:
                raise EventError(f"Event failed for rule {rule.name!r}.") from exc

            if not fires:
                continue

            try:
                model = rule.action(model, context)
            except PASSTHROUGH_EXCEPTIONS:
                raise
            except Exception as exc:
                raise ActionError(f"Action failed for rule {rule.name!r}.") from exc
        return model

    def advance(
        self,
        model: ModelT,
        *,
        time: Number,
        previous_time: Number = 0.0,
        unit_system: UnitSystem = UnitSystem.FIELD,
        runspec: RunSpec | None = None,
        state: object | None = None,
        time_step: int | None = None,
        previous_time_step: int | None = None,
        step_size: Number | None = None,
        extra: typing.Mapping[str, object] | None = None,
    ) -> ModelT:
        """
        Builds a `ScheduleContext` and calls `apply`.

        :param model: The model to advance.
        :param time: Elapsed time to advance to.
        :param previous_time: Elapsed time last advanced to.
        :param unit_system: Unit system for `time`/`previous_time`.
        :param runspec: The run's configuration, if any.
        :param state: The latest solved state, if any.
        :param time_step: The current time-step index, if tracked.
        :param previous_time_step: The time-step index last advanced at, if tracked.
        :param step_size: The current time-step's size, if tracked.
        :param extra: Additional domain-specific context.
        :returns: The model after every firing rule's action has run.
        """
        context = ScheduleContext(
            time=time,
            previous_time=previous_time,
            unit_system=unit_system,
            runspec=runspec,
            state=state,
            time_step=time_step,
            previous_time_step=previous_time_step,
            step_size=step_size,
            extra=extra or {},
        )
        return self.apply(model=model, context=context)

    def dump(self) -> dict[str, list[typing.Mapping[str, object]]]:
        """
        Dumps every rule.

        :returns: `{"rules": [...]}`. Every rule's `event`/`action` must
            be a `SerializableEvent`/`SerializableAction`.
        """
        return {"rules": [rule.dump() for rule in self.rules]}

    @classmethod
    def load(cls, data: typing.Mapping[str, typing.Sequence[typing.Mapping[str, object]]]) -> Self:
        """
        Loads a `Schedule` from `dump()`'s own output.

        :param data: `{"rules": [...]}`, as produced by `dump()`.
        :returns: The loaded `Schedule`.
        """
        return cls(rules=tuple(Rule.load(data=rule_data) for rule_data in data["rules"]))

    def __len__(self) -> int:
        """:returns: How many rules this schedule has."""
        return len(self.rules)

    def __iter__(self) -> typing.Iterator[Rule[ModelT]]:
        """:returns: An iterator over this schedule's rules, in order."""
        return iter(self.rules)

    def __contains__(self, name: str) -> bool:
        """:returns: Whether any rule in this schedule has this name."""
        return any(rule.name == name for rule in self.rules)

    def __add__(self, other: Self) -> Self:
        """
        Concatenates two schedules' rules, in order, keeping duplicates.

        :param other: The schedule to append.
        :returns: A new `Schedule` with `self`'s rules followed by `other`'s.
        """
        if not isinstance(other, Schedule):
            return NotImplemented
        return self.__class__(rules=self.rules + other.rules)

    def __or__(self, other: Self) -> Self:
        """
        Merges two schedules, `other`'s named rules overriding `self`'s
        own rules of the same name - the same override convention as
        `dict | dict`. Unnamed rules from both are kept, never merged.

        :param other: The schedule to merge in.
        :returns: A new, merged `Schedule`.
        """
        if not isinstance(other, Schedule):
            return NotImplemented
        other_names = {rule.name for rule in other.rules if rule.name is not None}
        kept = [rule for rule in self.rules if rule.name is None or rule.name not in other_names]
        return self.__class__(rules=tuple(kept) + other.rules)
