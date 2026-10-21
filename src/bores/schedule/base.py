"""Event-driven scheduling: `Event`/`Action` protocols, `ScheduleItem`, `Schedule`."""

import datetime
import math
import typing
from uuid import uuid4

import attrs
from typing_extensions import Self

from bores.errors import ActionError, EventError, StopSimulation, ValidationError
from bores.serde.base import Serializable
from bores.types import Boolean, Number, UnitSystem

__all__ = [
    "PASSTHROUGH_EXCEPTIONS",
    "Action",
    "Event",
    "Firings",
    "ModelT",
    "Schedule",
    "ScheduleContext",
    "ScheduleItem",
]

ModelT = typing.TypeVar("ModelT")
ModelTcon = typing.TypeVar("ModelTcon", contravariant=True)

PASSTHROUGH_EXCEPTIONS: tuple[type[BaseException], ...] = (StopSimulation,)
"""Exceptions `Schedule.apply` re-raises as they are, never wrapping them in `EventError`/`ActionError`."""


@attrs.define(kw_only=True, slots=True)
class Firings(Serializable):
    """
    How often, and when last, each limited `ScheduleItem` has fired. Create one per run, pass it
    to every `Schedule.advance` call, and save it with the run so a restart picks up the counts.
    """

    counts: dict[str, int] = attrs.field(factory=dict)
    """How many times each limited item, by name, has fired."""

    last_times: dict[str, Number] = attrs.field(factory=dict)
    """The elapsed time each limited item, by name, last fired at."""

    def count(self, name: str, /) -> int:
        """
        :param name: The item's name.
        :returns: How many times the item has fired.
        """
        return self.counts.get(name, 0)

    def last_time(self, name: str, /) -> Number | None:
        """
        :param name: The item's name.
        :returns: The elapsed time the item last fired at, or `None` if it never has.
        """
        return self.last_times.get(name)

    def record(self, *, name: str, time: Number) -> None:
        """
        Notes that an item has fired.

        :param name: The item's name.
        :param time: The elapsed time it fired at.
        """
        self.counts[name] = self.count(name) + 1
        self.last_times[name] = time


@attrs.frozen(kw_only=True, slots=True)
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

    firings: Firings | None = None
    """
    Counts how often limited items have fired. Required whenever the schedule has an item
    with a `max_firings` or `min_interval` limit.
    """


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


@attrs.frozen(kw_only=True, slots=True, repr=False, hash=True, unsafe_hash=True)
class ScheduleItem(typing.Generic[ModelT]):
    """
    One `(Event, Action)` pairing: `action` runs whenever `event` occurs.

    An event that holds across many checks, such as a rate threshold, would run its action at
    every check while it holds. `max_firings` and `min_interval` keep that from repeating a
    change that should happen once, like multiplying a connection factor.
    """

    event: Event[ModelT] = attrs.field(hash=False)
    """Decides whether and when `action` runs."""

    action: Action[ModelT] = attrs.field(hash=False)
    """Changes the model when `event` fires."""

    name: str = attrs.field(factory=lambda: uuid4().hex, validator=attrs.validators.min_len(1))
    """Optional label. If not provided, a random UUID is generated."""

    max_firings: Number | None = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(1))
    )
    """
    The most times `action` may run over a run. Leave it `None` for the default: once if
    `event` holds across checks (a threshold, for example), otherwise unlimited. Give
    `math.inf` to let such an event run its action at every check.
    """

    min_interval: Number = attrs.field(default=0.0, validator=attrs.validators.ge(0))
    """The least elapsed time allowed between two runs of `action`."""

    def __attrs_post_init__(self) -> None:
        if self.max_firings is None:
            held = getattr(self.event, "level_triggered", False)
            object.__setattr__(self, "max_firings", 1 if held else math.inf)

    @property
    def is_limited(self) -> Boolean:
        """Whether this item's runs are capped or spaced out, so `Firings` must track it."""
        return self.max_firings != math.inf or self.min_interval > 0

    def may_fire(self, *, firings: Firings, time: Number) -> Boolean:
        """
        :param firings: The run's record of past firings.
        :param time: The elapsed time being advanced to.
        :returns: Whether this item still has runs left and enough time has passed since its last.
        """
        if firings.count(self.name) >= typing.cast(Number, self.max_firings):
            return False
        last_time = firings.last_time(self.name)
        return last_time is None or (time - last_time) >= self.min_interval

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, when={self.event!r}, do={self.action!r})"


@attrs.frozen(slots=True)
class Schedule(typing.Generic[ModelT]):
    """Every item for a run. Advances a model by applying whichever items fire."""

    items: tuple[ScheduleItem[ModelT], ...] = attrs.field(converter=tuple, factory=tuple)
    """The `(Event, Action)` pairings to apply, in order."""

    _items_map: dict[str, ScheduleItem[ModelT]] = attrs.field(init=False, hash=False, repr=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "_items_map", {item.name: item for item in self.items})
        limited = [item.name for item in self.items if item.is_limited]
        repeated = sorted({name for name in limited if limited.count(name) > 1})
        if repeated:
            raise ValidationError(
                f"Items that limit how often they fire must have distinct names, but {repeated} "
                "each appear more than once. Give each its own `name`."
            )

    def apply(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Applies every item whose event fires, in `items` order.

        :param model: The model to advance.
        :param context: The current moment's context.
        :returns: The model after every firing item's action has run.
        :raises EventError: If an item's event raises anything other than
            a `PASSTHROUGH_EXCEPTIONS` member.
        :raises ActionError: If an item's action raises anything other
            than a `PASSTHROUGH_EXCEPTIONS` member.
        :raises ValidationError: If an item limits how often it fires and the context
            has no `firings`.
        """
        firings = context.firings
        for item in self.items:
            limited = item.is_limited
            if limited:
                if firings is None:
                    raise ValidationError(
                        f"Schedule item {item.name!r} limits how often it fires, so the context "
                        "must carry `firings` to count them. Create one `Firings()` per run and "
                        "pass it to every `advance` call."
                    )
                if not item.may_fire(firings=firings, time=context.time):
                    continue

            try:
                occurred = item.event(model, context)
            except PASSTHROUGH_EXCEPTIONS:
                raise
            except Exception as exc:
                raise EventError(f"Event failed for schedule item {item.name!r}.") from exc

            if not occurred:
                continue

            try:
                model = item.action(model, context)
            except PASSTHROUGH_EXCEPTIONS:
                raise
            except Exception as exc:
                raise ActionError(f"Action failed for schedule item {item.name!r}.") from exc

            if limited:
                firings.record(name=item.name, time=context.time)  # type: ignore[union-attr]
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
        firings: Firings | None = None,
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
        :param firings: The run's record of past firings. Required if any item limits how
            often it fires, which is the default for an item whose event holds across checks.
        :returns: The model after every firing item's action has run.
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
            firings=firings,
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
