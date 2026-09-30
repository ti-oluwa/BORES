"""Event-driven scheduling: `Event`/`Action` protocols, `Rule`, `Schedule`."""

import datetime
import threading
import typing
from uuid import uuid4

import attrs
from typing_extensions import Self

from bores.errors import ActionError, EventError, StopSimulation
from bores.serde.base import Serializable
from bores.serde.registry import (
    make_registry_deserializer,
    make_registry_serializer,
    make_serializable_type_registrar,
)
from bores.types import Boolean, Number, UnitSystem

__all__ = [
    "ACTION_TYPES",
    "EVENT_TYPES",
    "PASSTHROUGH_EXCEPTIONS",
    "Action",
    "Event",
    "ModelT",
    "Schedule",
    "ScheduleContext",
    "ScheduleItem",
    "SerializableAction",
    "SerializableEvent",
    "action_type",
    "event_type",
]

ModelT = typing.TypeVar("ModelT")
ModelTcon = typing.TypeVar("ModelTcon", contravariant=True)

PASSTHROUGH_EXCEPTIONS: tuple[type[BaseException], ...] = (StopSimulation,)
"""Exceptions `Schedule.apply` never wraps in `EventError`/`ActionError` - re-raised as is."""


@attrs.frozen(kw_only=True, slots=True, frozen=True)
class ScheduleContext:
    """Context passed to every `Event`/`Action` call alongside the model."""

    time: Number
    """Elapsed time the schedule is being advanced to, in `unit_system`."""

    previous_time: Number | None = None
    """
    Elapsed time the schedule was last advanced to. `None` on the first advance
    of a run, so an event at the very start (time zero) can still fire.
    """

    time_step: int | None = None
    """The current time-step index, if the caller is tracking one."""

    previous_time_step: int | None = None
    """The time-step index the schedule was last advanced at."""

    step_size: Number | None = None
    """The current time-step's size, in `unit_system`, if the caller is tracking one."""

    start_date: datetime.datetime | None = None
    """The simulation's start date, if the caller is tracking one."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `time`, `previous_time`, and any value an event reads."""

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


EVENT_TYPES: dict[str, type[SerializableEvent]] = {}
ACTION_TYPES: dict[str, type[SerializableAction]] = {}
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


dump_event = make_registry_serializer(
    base_cls=SerializableEvent, registry=EVENT_TYPES, key_attr="__type__"
)
load_event = make_registry_deserializer(base_cls=SerializableEvent, registry=EVENT_TYPES)
dump_action = make_registry_serializer(
    base_cls=SerializableAction, registry=ACTION_TYPES, key_attr="__type__"
)
load_action = make_registry_deserializer(base_cls=SerializableAction, registry=ACTION_TYPES)


@attrs.frozen(kw_only=True, slots=True, repr=False, hash=True, unsafe_hash=True)
class ScheduleItem(
    Serializable,
    typing.Generic[ModelT],
    serializers={"event": dump_event, "action": dump_action},
    deserializers={"event": load_event, "action": load_action},
):
    """One `(Event, Action)` pairing: `action` fires whenever `event` occurs."""

    event: Event[ModelT] = attrs.field(hash=False)
    """Decides whether and when `action` runs."""

    action: Action[ModelT] = attrs.field(hash=False)
    """Changes the model when `event` fires."""

    name: str = attrs.field(factory=lambda: uuid4().hex, validator=attrs.validators.min_len(1))
    """Optional label. If not provided, a random UUID is generated."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, when={self.event!r}, do={self.action!r})"


@attrs.frozen(slots=True)
class Schedule(
    Serializable,
    typing.Generic[ModelT],
    dump_exclude={"_items_map"},
    load_exclude={"_items_map"},
):
    """Every rule for a run. Advances a model by applying whichever items fire."""

    items: tuple[ScheduleItem[ModelT], ...] = attrs.field(converter=tuple, factory=tuple)
    """The `(Event, Action)` pairings to apply, in order."""

    _items_map: dict[str, ScheduleItem[ModelT]] = attrs.field(init=False, hash=False, repr=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "_items_map", {item.name: item for item in self.items})

    def apply(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Applies every rule whose event fires, in `items` order.

        :param model: The model to advance.
        :param context: The current moment's context.
        :returns: The model after every firing rule's action has run.
        :raises EventError: If a rule's event raises anything other than
            a `PASSTHROUGH_EXCEPTIONS` member.
        :raises ActionError: If a rule's action raises anything other
            than a `PASSTHROUGH_EXCEPTIONS` member.
        """
        for rule in self.items:
            try:
                occurred = rule.event(model, context)
            except PASSTHROUGH_EXCEPTIONS:
                raise
            except Exception as exc:
                raise EventError(f"Event failed for rule {rule.name!r}.") from exc

            if not occurred:
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
        previous_time: Number | None = None,
        unit_system: UnitSystem = UnitSystem.FIELD,
        time_step: int | None = None,
        previous_time_step: int | None = None,
        step_size: Number | None = None,
        start_date: datetime.datetime | None = None,
        extra: typing.Mapping[str, object] | None = None,
    ) -> ModelT:
        """
        Builds a `ScheduleContext` and calls `apply`.

        :param model: The model to advance.
        :param time: Elapsed time to advance to.
        :param previous_time: Elapsed time last advanced to. `None` on the first advance of a run.
        :param unit_system: Unit system for `time`/`previous_time`.
        :param time_step: The current time-step index, if tracked.
        :param previous_time_step: The time-step index last advanced at, if tracked.
        :param step_size: The current time-step's size, if tracked.
        :param start_date: The simulation's start date, if tracked.
        :param extra: Additional domain-specific context.
        :returns: The model after every firing rule's action has run.
        """
        context = ScheduleContext(
            time=time,
            previous_time=previous_time,
            unit_system=unit_system,
            time_step=time_step,
            previous_time_step=previous_time_step,
            step_size=step_size,
            start_date=start_date,
            extra=extra or {},
        )
        return self.apply(model=model, context=context)

    def get(self, name: str, /) -> ScheduleItem[ModelT] | None:
        """
        Returns the `ScheduleItem` with the given name, if any.

        :param name: The name of the item to retrieve.
        :returns: The item with that name, or `None` if not found.
        """
        return self._items_map.get(name)

    def __getitem__(self, name: str, /) -> ScheduleItem[ModelT]:
        """
        Returns the `ScheduleItem` with the given name, if any.

        :param name: The name of the item to retrieve.
        :returns: The item with that name.
        :raises KeyError: If no item with that name exists.
        """
        try:
            return self._items_map[name]
        except KeyError as exc:
            raise KeyError(f"No schedule item named {name!r}.") from exc

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> typing.Iterator[ScheduleItem[ModelT]]:
        return iter(self.items)

    def __contains__(self, o: typing.Any, /) -> bool:
        if isinstance(o, str):
            return o in self._items_map
        return o in self.items

    def __add__(self, other: Self | ScheduleItem[ModelT]) -> Self:
        """
        Concatenates two schedules' items, in order, keeping duplicates.

        :param other: The schedule to append.
        :returns: A new `Schedule` with `self`'s items followed by `other`'s.
        """
        if isinstance(other, ScheduleItem):
            other_items = (other,)
        elif isinstance(other, Schedule):
            other_items = other.items
        else:
            return NotImplemented
        return self.__class__(items=self.items + other_items)

    def __sub__(self, other: Self | ScheduleItem[ModelT]) -> Self:
        """
        Removes any items from `self` that have the same name as an item in `other`.

        :param other: The schedule whose item names to remove.
        :returns: A new `Schedule` with only items whose names are not in `other`.
        """
        if isinstance(other, ScheduleItem):
            other_names = {other.name}
        elif isinstance(other, Schedule):
            other_names = other._items_map
        else:
            return NotImplemented
        kept = tuple(item for item in self.items if item.name not in other_names)
        return self.__class__(items=kept)

    def __or__(self, other: Self | ScheduleItem[ModelT]) -> Self:
        """
        Merges two schedules, `other`'s named items overriding `self`'s
        own items of the same name. Unnamed items from both are kept, never merged.

        :param other: The schedule to merge in.
        :returns: A new, merged `Schedule`.
        """
        if isinstance(other, ScheduleItem):
            other_items = (other,)
            other_names = {other.name}
        elif isinstance(other, Schedule):
            other_items = other.items
            other_names = other._items_map
        else:
            return NotImplemented
        kept = tuple(item for item in self.items if item.name not in other_names)
        return self.__class__(items=kept + other_items)
