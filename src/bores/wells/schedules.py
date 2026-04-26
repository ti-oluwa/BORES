import itertools
import threading
import typing

import attrs
from typing_extensions import Self

from bores.errors import DeserializationError, SerializationError, ValidationError
from bores.serialization import Serializable
from bores.states import ModelState
from bores.stores import StoreSerializable
from bores.types import Coordinates, S, T
from bores.wells.base import InjectionWell, ProductionWell, Well, Wells, WellT
from bores.wells.controls import WellControl
from bores.wells.core import InjectedFluid, ProducedFluid, WellFluid
from bores.wells.indices import update_wells_indices

__all__ = [
    "EventAction",
    "EventActions",
    "EventPredicate",
    "EventPredicates",
    "WellEvent",
    "WellSchedule",
    "WellSchedules",
    "event_action",
    "event_predicate",
    "time_predicate",
    "update_well",
]


PredicateFunc = typing.Callable[[S, T], bool]
"""A function that takes two arguments of types S and T and returns a boolean value."""
ActionFunc = typing.Callable[[S, T], None]
"""A function that takes two arguments of types S and T and returns None."""


class EventPredicates(Serializable, typing.Generic[WellT, Coordinates]):
    """Composite predicate that chains multiple predicate functions."""

    __abstract_serializable__ = True

    def __init__(
        self,
        *predicates: PredicateFunc[WellT, ModelState[Coordinates]],
        on_any: bool = False,
    ):
        """
        Initializes the `EventPredicates` with a sequence of predicate functions.

        :param predicates: A sequence of predicate functions to be chained.
        :param on_any: If True, the composite predicate returns True if any of the predicates return True.
                       If False, it returns True only if all predicates return True.
        """
        self.predicates = predicates
        self.on_any = on_any

    def __call__(self, well: WellT, state: ModelState[Coordinates]) -> bool:
        """
        Calls the composite predicate function.

        :param well: The well to which this predicate is applied.
        :param state: The current model state in the simulation.
        :return: A boolean indicating whether to apply the event.
        """
        results = (predicate(well, state) for predicate in self.predicates)
        return any(results) if self.on_any else all(results)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the composite predicate."""
        return {
            "predicates": [
                serialize_event_predicate(predicate, recurse)
                for predicate in self.predicates
            ],
            "on_any": self.on_any,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Deserialize the composite predicate."""
        if "predicates" not in data or "on_any" not in data:
            raise DeserializationError(
                "Invalid data for well predicates deserialization"
            )

        predicates = [
            deserialize_event_predicate(predicate_data)
            for predicate_data in data["predicates"]
        ]
        return cls(*predicates, on_any=data["on_any"])


class EventActions(Serializable, typing.Generic[WellT, Coordinates]):
    """Composite action that chains multiple action functions."""

    __abstract_serializable__ = True

    def __init__(self, *actions: ActionFunc[WellT, ModelState[Coordinates]]):
        """
        Initializes the `EventActions` with a sequence of action functions.

        :param actions: A sequence of action functions to be chained.
        """
        if not actions:
            raise ValidationError("At least one action must be provided to chain.")
        self.actions = actions

    def __call__(self, well: WellT, state: ModelState[Coordinates]) -> None:
        """
        Executes all chained action functions in sequence.

        :param well: The well to which the actions are applied.
        :param state: The current model state in the simulation.
        """
        for action in self.actions:
            action(well, state)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the composite action."""
        return {
            "actions": [
                serialize_event_action(action, recurse) for action in self.actions
            ]
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Deserialize the composite action."""
        if "actions" not in data:
            raise DeserializationError("Invalid data for well actions deserialization")

        actions = [
            deserialize_event_action(action_data) for action_data in data["actions"]
        ]
        return cls(*actions)


@attrs.define(slots=True)
class EventPredicate(Serializable, typing.Generic[WellT, Coordinates]):
    """A predicate that can be serialized and composed with complex expressions."""

    _func: typing.Optional[PredicateFunc[WellT, ModelState[Coordinates]]] = attrs.field(
        default=None, repr=False
    )
    """The underlying predicate function. If None, this is a composite predicate."""
    _op: typing.Optional[typing.Literal["and", "or", "not"]] = attrs.field(default=None)
    """The logical operation for composite predicates."""
    _operands: typing.Tuple[Self, ...] = attrs.field(factory=tuple)
    """The operand predicates for composite predicates."""

    @classmethod
    def from_func(cls, func: PredicateFunc[WellT, ModelState[Coordinates]]) -> Self:
        """Create an EventPredicate from a function."""
        return cls(_func=func)

    @classmethod
    def from_event_predicates(
        cls, event_predicates: EventPredicates[WellT, Coordinates]
    ) -> Self:
        """
        Create `EventPredicate` from `EventPredicates` for interoperability.

        :param event_predicates: The `EventPredicates` instance to convert.
        :return: An `EventPredicate` instance.
        """
        return cls(_func=event_predicates)

    @classmethod
    def any_of(cls, *predicates: PredicateFunc[WellT, ModelState[Coordinates]]) -> Self:
        """Convenience method for creating logical OR of multiple predicates."""
        if len(predicates) == 1:
            return cls.from_func(predicates[0])
        result = cls.from_func(predicates[0])
        for pred in predicates[1:]:
            result = result | pred
        return result

    @classmethod
    def all_of(cls, *predicates: PredicateFunc[WellT, ModelState[Coordinates]]) -> Self:
        """Convenience method for creating logical AND of multiple predicates."""
        if len(predicates) == 1:
            return cls.from_func(predicates[0])
        result = cls.from_func(predicates[0])
        for pred in predicates[1:]:
            result = result & pred
        return result

    def __call__(self, well: WellT, state: ModelState[Coordinates]) -> bool:
        if self._func is not None:
            # Leaf node, call the actual predicate function
            return self._func(well, state)
        elif self._op == "and":
            return all(op(well, state) for op in self._operands)  # type: ignore
        elif self._op == "or":
            return any(op(well, state) for op in self._operands)  # type: ignore
        elif self._op == "not":
            return not self._operands[0](well, state)  # type: ignore
        raise ValueError("Invalid EventPredicate state: no func or op defined")

    def __and__(
        self, other: typing.Union[Self, PredicateFunc[WellT, ModelState[Coordinates]]]
    ) -> Self:
        """Combine with AND logic: self & other"""
        if not isinstance(other, EventPredicate):
            other = EventPredicate.from_func(other)
        return self.__class__(_op="and", _operands=(self, other))  # type: ignore

    def __or__(
        self, other: typing.Union[Self, PredicateFunc[WellT, ModelState[Coordinates]]]
    ) -> Self:
        """Combine with OR logic: self | other"""
        if not isinstance(other, EventPredicate):
            other = EventPredicate.from_func(other)
        return self.__class__(_op="or", operands=(self, other))  # type: ignore

    def __invert__(self) -> Self:
        """Invert with NOT logic: ~self"""
        return self.__class__(_op="not", _operands=(self,))  # type: ignore

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the predicate expression."""
        if self._func is not None:
            # Leaf node, serialize the underlying function
            return {
                "type": "event_predicate_leaf",
                "func": serialize_event_predicate(self._func, recurse),
            }
        # Composite node, serialize the expression tree
        return {
            "type": "event_predicate_composite",
            "op": self._op,
            "operands": [op.dump(recurse) for op in self._operands],  # type: ignore
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Deserialize the predicate expression."""
        pred_type = data.get("type")

        if pred_type == "event_predicate_leaf":
            # Deserialize leaf node
            func = deserialize_event_predicate(data["func"])
            return cls(_func=func)
        elif pred_type == "event_predicate_composite":
            # Deserialize composite node
            operands = tuple(cls.load(op_data) for op_data in data["operands"])
            return cls(_op=data["op"], _operands=operands)  # type: ignore
        raise DeserializationError(f"Unknown {cls.__name__} type: {pred_type}")


@attrs.define(slots=True)
class EventAction(Serializable, typing.Generic[WellT, Coordinates]):
    """An action that can be serialized and composed."""

    _func: typing.Optional[ActionFunc[WellT, ModelState[Coordinates]]] = attrs.field(
        default=None, repr=False
    )
    """The underlying action function. If None, this is a composite action."""
    _actions: typing.Tuple[Self, ...] = attrs.field(factory=tuple)
    """The chained actions for composite actions."""

    @classmethod
    def from_func(cls, func: ActionFunc[WellT, ModelState[Coordinates]]) -> Self:
        """Create an `EventAction` from a function."""
        return cls(_func=func)

    @classmethod
    def from_actions(cls, actions: EventActions[WellT, Coordinates]) -> Self:
        """Create EventAction from EventActions for interoperability."""
        return cls(_func=actions)

    @classmethod
    def sequence(cls, *actions: ActionFunc[WellT, ModelState[Coordinates]]) -> Self:
        """Convenience: create sequence of actions."""
        if len(actions) == 1:
            return cls.from_func(actions[0])

        result = cls.from_func(actions[0])
        for action in actions[1:]:
            result = result & action
        return result

    def __call__(self, well: WellT, state: ModelState[Coordinates]) -> None:
        if self._func is not None:
            # Leaf node, call the actual action function
            self._func(well, state)
        else:
            # Execute all actions in sequence
            for action in self._actions:
                action(well, state)  # type: ignore

    def __and__(
        self, other: typing.Union[Self, ActionFunc[WellT, ModelState[Coordinates]]]
    ) -> Self:
        """Chain actions: self & other (execute both in sequence)"""
        if not isinstance(other, EventAction):
            other = EventAction.from_func(other)

        # Flatten nested chains for efficiency
        left_actions = self._actions if self._func is None else (self,)
        right_actions = other._actions if other._func is None else (other,)
        return self.__class__(_actions=left_actions + right_actions)  # type: ignore

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the action."""
        if self._func is not None:
            # Leaf node, serialize the underlying function
            return {
                "type": "event_action_leaf",
                "func": serialize_event_action(self._func, recurse),
            }
        # Composite node, serialize the action chain
        return {
            "type": "event_action_composite",
            "actions": [action.dump(recurse) for action in self._actions],  # type: ignore
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Deserialize the action."""
        action_type = data.get("type")

        if action_type == "event_action_leaf":
            # Deserialize leaf node
            func = deserialize_event_action(data["func"])
            return cls(_func=func)
        elif action_type == "event_action_composite":
            # Deserialize composite node
            actions = tuple(cls.load(action_data) for action_data in data["actions"])
            return cls(_actions=actions)  # type: ignore
        raise DeserializationError(f"Unknown EventAction type: {action_type}")


_HOOKS: typing.Dict[str, PredicateFunc[Well, ModelState]] = {}
_ACTIONS: typing.Dict[str, ActionFunc[Well, ModelState]] = {}
_predicate_lock = threading.Lock()
_action_lock = threading.Lock()


@typing.overload
def event_predicate(
    func: PredicateFunc[WellT, ModelState[Coordinates]],
) -> PredicateFunc[WellT, ModelState[Coordinates]]: ...


@typing.overload
def event_predicate(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[
    [PredicateFunc[WellT, ModelState[Coordinates]]],
    PredicateFunc[WellT, ModelState[Coordinates]],
]: ...


@typing.overload
def event_predicate(
    func: PredicateFunc[WellT, ModelState[Coordinates]],
    name: typing.Optional[str] = None,
    override: bool = False,
) -> PredicateFunc[WellT, ModelState[Coordinates]]: ...


def event_predicate(
    func: typing.Optional[PredicateFunc] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[
    PredicateFunc[WellT, ModelState[Coordinates]],
    typing.Callable[
        [PredicateFunc[WellT, ModelState[Coordinates]]],
        PredicateFunc[WellT, ModelState[Coordinates]],
    ],
]:
    """
    Register a well predicate function for serialization.

    A well predicate is a callable that takes a well and the model state as arguments
    and returns a boolean indicating whether to apply a scheduled event.

    :param func: The predicate function to register.
    :param name: The name to register the predicate under. If None, the function's
        `__name__` attribute is used.
    :param override: If True, override any existing predicate with the same name.
    :return: The registered predicate function or a decorator to register the function.
    """

    def decorator(func: PredicateFunc) -> PredicateFunc:
        predicate_name = name or getattr(func, "__name__", None)
        if not predicate_name:
            raise ValidationError(
                "Hook function must have a `__name__` attribute or a name must be provided."
            )

        with _predicate_lock:
            if predicate_name in _HOOKS and not override:
                raise ValidationError(
                    f"Hook '{predicate_name}' already registered. Use `override=True` or provide a different name."
                )

            _HOOKS[predicate_name] = func
        return func

    if func is not None:
        return decorator(func)
    return decorator


@typing.overload
def event_action(
    func: ActionFunc[WellT, ModelState[Coordinates]],
) -> ActionFunc[WellT, ModelState[Coordinates]]: ...


@typing.overload
def event_action(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[
    [ActionFunc[WellT, ModelState[Coordinates]]],
    ActionFunc[WellT, ModelState[Coordinates]],
]: ...


@typing.overload
def event_action(
    func: ActionFunc[WellT, ModelState[Coordinates]],
    name: typing.Optional[str] = None,
    override: bool = False,
) -> ActionFunc[WellT, ModelState[Coordinates]]: ...


def event_action(
    func: typing.Optional[ActionFunc[WellT, ModelState[Coordinates]]] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[
    ActionFunc[WellT, ModelState[Coordinates]],
    typing.Callable[
        [ActionFunc[WellT, ModelState[Coordinates]]],
        ActionFunc[WellT, ModelState[Coordinates]],
    ],
]:
    """
    Register a well action function for serialization.

    A well action is a callable that takes a well and the model state as arguments
    and performs the event action.

    :param func: The action function to register.
    :param name: The name to register the action under. If None, the function's
        __name__ attribute is used.
    :param override: If True, override any existing action with the same name.
    :return: The registered action function or a decorator to register the function.
    """

    def decorator(func: ActionFunc) -> ActionFunc:
        action_name = name or getattr(func, "__name__", None)
        if not action_name:
            raise ValidationError(
                "Action function must have a `__name__` attribute or a name must be provided."
            )
        with _action_lock:
            if action_name in _ACTIONS and not override:
                raise ValidationError(
                    f"Action '{action_name}' already registered. Use `override=True` or provide a different name."
                )

            _ACTIONS[action_name] = func
        return func

    if func is not None:
        return decorator(func)
    return decorator


def list_event_predicates() -> typing.List[str]:
    """List all registered well event predicates."""
    with _predicate_lock:
        return list(_HOOKS.keys())


def list_event_actions() -> typing.List[str]:
    """List all registered well event actions."""
    with _action_lock:
        return list(_ACTIONS.keys())


def get_event_predicate(name: str) -> PredicateFunc[Well, ModelState]:
    """
    Get a registered well event predicate by name.

    :param name: The name of the registered predicate.
    :return: The predicate function associated with the given name.
    :raises ValidationError: If the predicate is not registered.
    """
    with _predicate_lock:
        if name not in _HOOKS:
            raise ValidationError(
                f"Hook '{name}' not registered. Use `@event_predicate` to register it."
            )
        return _HOOKS[name]


def get_event_action(name: str) -> ActionFunc[Well, ModelState]:
    """
    Get a registered well event action by name.

    :param name: The name of the registered action.
    :return: The action function associated with the given name.
    :raises ValidationError: If the action is not registered.
    """
    with _action_lock:
        if name not in _ACTIONS:
            raise ValidationError(
                f"Action '{name}' not registered. Use `@event_action` to register it."
            )
        return _ACTIONS[name]


def serialize_event_predicate(
    predicate: PredicateFunc[WellT, ModelState], recurse: bool = True
) -> typing.Dict[str, typing.Any]:
    """
    Serialize a well event predicate function.

    :param predicate: The predicate function to serialize.
    :param recurse: Whether to recursively serialize nested predicates.
    :return: A dictionary representing the serialized predicate.
    """
    if isinstance(predicate, EventPredicate):
        return predicate.dump(recurse)

    # Check for registered predicates
    with _predicate_lock:
        for name, registered_predicate in _HOOKS.items():
            if predicate is registered_predicate:
                return {"type": "registered", "name": name}

    # Check for built-in predicate types
    if isinstance(predicate, TimePredicate):
        return {
            "type": "time_predicate",
            "data": predicate.dump(recurse),
        }

    if isinstance(predicate, EventPredicates):
        return {
            "type": "composite_predicates",
            "data": predicate.dump(recurse),
        }

    raise SerializationError(
        f"Cannot serialize predicate {predicate}. Please register it with `@event_predicate`."
    )


def deserialize_event_predicate(
    data: typing.Mapping[str, typing.Any],
) -> PredicateFunc[Well, ModelState]:
    """
    Deserialize a well event predicate function.

    :param data: The serialized predicate data.
    :return: The deserialized predicate function.
    """
    if "type" not in data:
        raise DeserializationError("Invalid data for predicate deserialization")

    predicate_type = data["type"]
    if predicate_type in ("event_predicate_leaf", "event_predicate_composite"):
        return EventPredicate.load(data)

    if predicate_type == "registered":
        with _predicate_lock:
            if data["name"] not in _HOOKS:
                raise DeserializationError(f"Hook '{data['name']}' not registered")
            return _HOOKS[data["name"]]

    elif predicate_type == "time_predicate":
        if "data" not in data:
            raise DeserializationError(
                "Invalid data for time predicate deserialization"
            )
        return TimePredicate.load(data["data"])

    elif predicate_type == "composite_predicates":
        if "data" not in data:
            raise DeserializationError(
                "Invalid data for composite predicates deserialization"
            )
        return EventPredicates.load(data["data"])

    raise DeserializationError(
        f"Unknown predicate type: {predicate_type}. Please register it with `@event_predicate`."
    )


def serialize_event_action(
    action: ActionFunc[WellT, ModelState], recurse: bool = True
) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """
    Serialize a well event action function.

    :param action: The action function to serialize.
    :param recurse: Whether to recursively serialize nested actions.
    :return: A dictionary representing the serialized action.
    """
    if isinstance(action, EventAction):
        return action.dump(recurse)

    # Check for registered actions
    with _action_lock:
        for name, registered_action in _ACTIONS.items():
            if action is registered_action:
                return {"type": "registered", "name": name}

    if isinstance(action, UpdateAction):
        return {
            "type": "update_action",
            "data": action.dump(recurse),
        }

    if isinstance(action, EventActions):
        return {
            "type": "composite_actions",
            "data": action.dump(recurse),
        }

    raise SerializationError(
        f"Cannot serialize action {action}. Please register it with `@event_action`."
    )


def deserialize_event_action(
    data: typing.Mapping[str, typing.Any],
) -> ActionFunc[Well, ModelState]:
    """
    Deserialize a well event action function.

    :param data: The serialized action data.
    :return: The deserialized action function.
    """
    if "type" not in data:
        raise DeserializationError("Invalid data for action deserialization")

    action_type = data["type"]
    if action_type in ("event_action_leaf", "event_action_composite"):
        return EventAction.load(data)

    if action_type == "registered":
        with _action_lock:
            if data["name"] not in _ACTIONS:
                raise DeserializationError(f"Action '{data['name']}' not registered")
            return _ACTIONS[data["name"]]

    elif action_type == "update_action":
        if "data" not in data:
            raise DeserializationError("Invalid data for update action deserialization")
        return UpdateAction.load(data["data"])

    elif action_type == "composite_actions":
        if "data" not in data:
            raise DeserializationError(
                "Invalid data for composite actions deserialization"
            )
        return EventActions.load(data["data"])

    raise DeserializationError(
        f"Unknown action type: {action_type}. Please register it with `@event_action`."
    )


@typing.final
@attrs.define(slots=True, hash=True)
class WellEvent(Serializable, typing.Generic[Coordinates]):
    """
    Represents a scheduled event for a well at a specific time step.

    This event can include changes to the well's bottom-hole pressure, skin factor,
    and whether the well is active or not.
    The event is applied to the well at the specified time step.
    """

    predicate: PredicateFunc[Well[Coordinates, WellFluid], ModelState[Coordinates]]
    """A callable predicate that takes the well and model state as arguments and returns a boolean indicating whether to apply the event."""

    action: ActionFunc[Well[Coordinates, WellFluid], typing.Any]
    """A callable action that takes the well and model state as arguments and performs the event action."""

    def __call__(
        self, well: Well[Coordinates, WellFluid], state: ModelState[Coordinates]
    ) -> None:
        """
        Apply this schedule to a well.

        :param well: The well to which this schedule will be applied.
        :param state: The current model state in the simulation.
        """
        self.action(well, state)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the well event."""
        return {
            "predicate": serialize_event_predicate(self.predicate, recurse),
            "action": serialize_event_action(self.action, recurse),
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> "WellEvent":
        """Deserialize the well event."""
        if "predicate" not in data or "action" not in data:
            raise DeserializationError("Invalid data for well event deserialization")

        return cls(
            predicate=deserialize_event_predicate(data["predicate"]),
            action=deserialize_event_action(data["action"]),
        )


@typing.final
@attrs.frozen(slots=True)
class WellSchedule(Serializable, typing.Generic[Coordinates]):
    """
    A collection of scheduled events for a well.

    Each event is identified by a unique string identifier.

    Example:

    ```python
    schedule = WellSchedule()
    schedule.add("event1", event=WellEvent(...))
    ```
    """

    events: typing.Dict[str, WellEvent[Coordinates]] = attrs.field(
        factory=dict, init=False
    )
    """A dictionary mapping time steps to scheduled events."""
    _events_hashes: typing.Dict[int, str] = attrs.field(factory=dict, init=False)
    """A set to track hashes of added events for uniqueness."""

    def add(self, id_: str, /, event: WellEvent[Coordinates]) -> None:
        """
        Adds an event to the schedule.

        :param event: The event to add.
        :param id: The unique identifier for the event.
        """
        self.events[id_] = event
        self._events_hashes[hash(event)] = id_

    def remove(self, o: typing.Union[WellEvent[Coordinates], str], /) -> None:
        """
        Removes an event from the schedule if it exists.

        :param o: The event or its identifier to remove.
        """
        if isinstance(o, WellEvent):
            # If o is a WellEvent, find its ID and remove it
            event_hash = hash(o)
            if event_hash in self._events_hashes:
                del self.events[self._events_hashes[event_hash]]
                del self._events_hashes[event_hash]
        else:
            # If o is an ID, remove the event with that ID
            if o in self.events:
                event_hash = hash(self.events[o])
                del self._events_hashes[event_hash]
                del self.events[o]

    def clear(self) -> None:
        """Clears all events from the schedule."""
        self.events.clear()
        self._events_hashes.clear()

    def get(self, id_: str, /) -> typing.Optional[WellEvent[Coordinates]]:
        """
        Retrieves an event from the schedule by its identifier.

        :param id: The unique identifier for the event.
        :return: The event if found, None otherwise.
        """
        return self.events.get(id_, None)

    def __getitem__(self, id_: str, /) -> WellEvent[Coordinates]:
        """
        Retrieves an event from the schedule by its identifier.

        :param id: The unique identifier for the event.
        :return: The event associated with the given identifier.
        :raises KeyError: If the identifier is not found in the schedule.
        """
        try:
            return self.events[id_]
        except KeyError:
            raise KeyError(f"Event with id {id_} not found")

    def __setitem__(self, id_: str, event: WellEvent[Coordinates], /) -> None:
        """
        Sets an event in the schedule with the given identifier.

        :param id: The unique identifier for the event.
        :param event: The event to set.
        """
        self.events[id_] = event
        self._events_hashes[hash(event)] = id_

    def __contains__(self, o: typing.Union[WellEvent, str], /) -> bool:
        """
        Checks if an event or its identifier is in the schedule.

        :param o: The event or its identifier to check.
        :return: True if the event or its identifier is in the schedule, False otherwise.
        """
        if isinstance(o, WellEvent):
            return hash(o) in self._events_hashes
        return o in self.events

    def __len__(self) -> int:
        """Returns the number of events in the schedule."""
        return len(self.events)

    def __iter__(self) -> typing.Iterator[str]:
        """Returns an iterator over the event identifiers in the schedule."""
        return iter(self.events)

    def apply(
        self,
        well: Well[Coordinates, WellFluid],
        state: ModelState[Coordinates],
        *ids: str,
    ) -> None:
        """
        Applies the schedule to a well.

        :param well: The well to which the schedule will be applied.
        :param state: The current model state in the simulation.
        :param ids: The unique identifiers of the events to apply.
        """
        if not ids:
            for event in self.events.values():
                if event.predicate(well, state):
                    event(well, state)

        else:
            for id_ in ids:
                event = self.events.get(id_, None)
                if event is None:
                    continue
                if event.predicate(well, state):
                    event(well, state)
        return None

    def __and__(self, other: Self) -> Self:
        combined_schedule = self.__class__()
        for id_, event in itertools.chain(self.events.items(), other.events.items()):
            combined_schedule.add(id_, event)
        return combined_schedule

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "events": {id_: event.dump(recurse) for id_, event in self.events.items()}
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        if "events" not in data:
            raise DeserializationError("Invalid data for well schedule deserialization")

        events = {
            id_: WellEvent.load(event_data)
            for id_, event_data in data["events"].items()
        }
        schedule = cls()
        for id_, event in events.items():
            schedule.add(id_, event)
        return schedule


@typing.final
@attrs.define(slots=True)
class WellSchedules(StoreSerializable, typing.Generic[Coordinates]):
    """
    A collection of schedules for multiple wells.

    Each well is identified by its name, and has an associated `WellSchedule`.

    Example:
    ```python
    schedule = WellSchedule()
    schedule.add("event1", WellEvent(...))

    schedules = WellSchedules()
    schedules.add("Well-A", schedule)
    ```
    """

    schedules: typing.Dict[str, WellSchedule[Coordinates]] = attrs.field(
        factory=dict, init=False
    )
    """A dictionary mapping well names to their schedules."""

    def add(self, well_name: str, schedule: WellSchedule[Coordinates]) -> None:
        """
        Adds a schedule for a given well.

        :param well_name: The name of the well.
        :param schedule: The schedule for the well.
        """
        self.schedules[well_name] = schedule

    def get(self, well_name: str) -> typing.Optional[WellSchedule[Coordinates]]:
        """
        Retrieves the schedule for a given well by its name.

        :param well_name: The name of the well.
        :return: The schedule for the well if found, None otherwise.
        """
        return self.schedules.get(well_name, None)

    def remove(self, well_name: str) -> None:
        """
        Removes the schedule for a given well by its name.

        :param well_name: The name of the well.
        """
        if well_name in self.schedules:
            del self.schedules[well_name]

    def clear(self) -> None:
        """Clears all well schedules."""
        self.schedules.clear()

    def __len__(self) -> int:
        """Returns the number of well schedules."""
        return len(self.schedules)

    def __iter__(self) -> typing.Iterator[str]:
        """Returns an iterator over the well names in the schedules."""
        return iter(self.schedules)

    def __contains__(self, well_name: str) -> bool:
        """
        Checks if a schedule for a given well name exists.

        :param well_name: The name of the well.
        :return: True if the schedule exists, False otherwise.
        """
        return well_name in self.schedules

    def __setitem__(self, well_name: str, schedule: WellSchedule[Coordinates]) -> None:
        """
        Sets the schedule for a given well name.

        :param well_name: The name of the well.
        :param schedule: The schedule for the well.
        """
        self.schedules[well_name] = schedule

    def __getitem__(self, well_name: str) -> WellSchedule[Coordinates]:
        """
        Retrieves the schedule for a given well name.

        :param well_name: The name of the well.
        :return: The schedule for the well.
        :raises KeyError: If the well name is not found in the schedules.
        """
        try:
            return self.schedules[well_name]
        except KeyError:
            raise KeyError(f"Schedule for well {well_name} not found")

    def apply(self, wells: Wells[Coordinates], state: ModelState[Coordinates]) -> None:
        """
        Applies all well schedules to their respective wells.

        :param wells: The collection of wells to which the schedules will be applied.
        :param state: The current model state in the simulation.
        """
        for well_name, schedule in self.schedules.items():
            well_list = wells.get_by_name(well_name)
            if not well_list:
                continue
            for well in well_list:
                if well is None:
                    continue
                schedule.apply(well, state)  # type: ignore
        return None

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "schedules": {
                well_name: schedule.dump(recurse)
                for well_name, schedule in self.schedules.items()
            }
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        if "schedules" not in data:
            raise DeserializationError(
                "Invalid data for well schedules deserialization"
            )

        schedules = {
            well_name: WellSchedule.load(schedule_data)
            for well_name, schedule_data in data["schedules"].items()
        }
        well_schedules = cls()
        for well_name, schedule in schedules.items():
            well_schedules.add(well_name, schedule)
        return well_schedules


class TimePredicate(
    Serializable,
    fields={"time_step": typing.Optional[int], "time": typing.Optional[float]},
):
    def __init__(
        self,
        time_step: typing.Optional[int] = None,
        time: typing.Optional[float] = None,
    ):
        """
        Initializes the well time predicate with either a specific time step or time.

        :param time_step: The specific time step at which to trigger the event.
        :param time: The specific simulation time at which to trigger the event.
        """
        if not (time_step or time):
            raise ValidationError("Either time_step or time must be provided.")
        self.time_step = time_step
        self.time = time
        self._triggered = False  # Internal state to prevent duplicate triggers

    def __call__(self, well: Well, state: ModelState[Coordinates]) -> bool:
        """
        The predicate function that checks if the event should be applied based on the model state.

        :param well: The well to which this predicate is applied.
        :param state: The current model state in the simulation.
        :return: A boolean indicating whether to apply the event.
        """
        # If already triggered, don't trigger again
        if self._triggered:
            return False

        # Check time step threshold
        if self.time_step is not None and state.step >= self.time_step:
            self._triggered = True
            return True

        # Check simulation time threshold (use >= to handle adaptive time stepping)
        if self.time is not None and state.time >= self.time:
            self._triggered = True
            return True

        return False


class UpdateAction(
    Serializable,
    fields={
        "control": typing.Optional[WellControl],
        "skin_factor": typing.Optional[float],
        "is_active": typing.Optional[bool],
        "injected_fluid": typing.Optional[InjectedFluid],
        "produced_fluids": typing.Optional[typing.Sequence[ProducedFluid]],
    },
):
    def __init__(
        self,
        control: typing.Optional[WellControl] = None,
        skin_factor: typing.Optional[float] = None,
        is_active: typing.Optional[bool] = None,
        injected_fluid: typing.Optional[InjectedFluid] = None,
        produced_fluids: typing.Optional[typing.Sequence[ProducedFluid]] = None,
    ):
        """
        Initializes the well update action with properties to update.

        :param control: New control strategy for the well.
        :param skin_factor: New skin factor for the well.
        :param is_active: New active status for the well (True for open, False for shut in).
        :param injected_fluid: New fluid properties for injection wells.
        :param produced_fluids: New fluid properties for production wells.
        """
        valid = any(
            param is not None
            for param in [
                control,
                skin_factor,
                is_active,
                injected_fluid,
                produced_fluids,
            ]
        )
        if not valid:
            raise ValidationError("At least one property must be provided to update.")

        self.control = control
        self.skin_factor = skin_factor
        self.is_active = is_active
        self.injected_fluid = injected_fluid
        self.produced_fluids = produced_fluids

    def __call__(self, well: Well, state: ModelState[Coordinates]) -> None:
        """
        The action function that modifies well configuration.

        :param well: The well to which this action is applied.
        :param state: The current model state in the simulation.
        """
        if self.control is not None:
            well.control = self.control

        if self.skin_factor is not None:
            skin_factor_changed = well.skin_factor != self.skin_factor
            well.skin_factor = self.skin_factor
            if skin_factor_changed:
                # Skin factor change warrants well indices update
                update_wells_indices.set(True)

        if self.is_active is True:
            well.open()
        elif self.is_active is False:
            well.shut_in()

        if self.injected_fluid is not None and isinstance(well, InjectionWell):
            well.injected_fluid = self.injected_fluid
        if self.produced_fluids is not None and isinstance(well, ProductionWell):
            well.produced_fluids = self.produced_fluids
        return


def time_predicate(
    time_step: typing.Optional[int] = None,
    time: typing.Optional[float] = None,
) -> PredicateFunc[Well, ModelState]:
    """
    Returns a predicate function that triggers at a specific time step or time.

    :param time_step: The specific time step at which to trigger the event.
    :param time: The specific simulation time at which to trigger the event.
    :return: A predicate function that takes a well and model state as arguments and returns a boolean indicating whether to apply the event.
    """
    return TimePredicate(time_step=time_step, time=time)


def update_well(
    control: typing.Optional[WellControl] = None,
    skin_factor: typing.Optional[float] = None,
    is_active: typing.Optional[bool] = None,
    injected_fluid: typing.Optional[InjectedFluid] = None,
    produced_fluids: typing.Optional[typing.Sequence[ProducedFluid]] = None,
) -> ActionFunc[Well, ModelState]:
    """
    Returns an action function that modifies well configuration.

    :param control: New control strategy for the well.
    :param skin_factor: New skin factor for the well.
    :param is_active: New active status for the well (True for open, False for shut in).
    :param injected_fluid: New fluid properties for injection wells.
    :param produced_fluids: New fluid properties for production wells.
    :return: An action function that takes a well and model state as arguments and performs the property updates.
    """
    return UpdateAction(
        control=control,
        skin_factor=skin_factor,
        is_active=is_active,
        injected_fluid=injected_fluid,
        produced_fluids=produced_fluids,
    )
