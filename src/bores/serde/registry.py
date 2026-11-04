import threading
import typing
from collections.abc import Mapping

from bores.errors import DeserializationError, ValidationError
from bores.serde.base import (
    Serializable,
    SerializableT,
    register_type_deserializer,
    register_type_serializer,
)

SerializableTo = typing.TypeVar("SerializableTo", bound=Serializable)


def make_serializable_type_registrar(
    base_cls: type[SerializableT],
    registry: dict[str, type[SerializableT]],
    key_attr: str = "__type__",
    lock: typing.Optional[threading.Lock] = None,  # noqa: UP045
    key_factory: typing.Callable[[type[SerializableT]], str] | None = None,
    override: bool = False,
    auto_register_serializer: bool = True,
    auto_register_deserializer: bool = True,
) -> typing.Callable[[type[SerializableTo]], type[SerializableTo]]:
    """
    Decorator factory to create a registrar for `Serializable` subclasses.

    Creates a decorator to register `Serializable` subclasses in a registry.

    :param base_cls: The base class for the `Serializable` subclasses.
    :param registry: The registry to store the subclasses.
    :param key_attr: The attribute to use as the registry key.
    :param lock: An optional lock for thread safety.
    :param key_factory: An optional factory function to generate the registry key.
    :param override: Whether to allow overriding existing registrations.
    :return: A decorator to register `Serializable` subclasses.
    """
    if not key_attr:
        raise ValidationError("`key_attr` must be a non-empty string.")

    lock = lock or threading.Lock()

    def registrar(cls: type[SerializableTo]) -> type[SerializableTo]:
        """Decorator to register a `Serializable` subclass."""
        if not issubclass(cls, base_cls):
            raise ValidationError(f"Class {cls!r} is not a subclass of {base_cls!r}")

        key = getattr(cls, key_attr, None)
        if not key:
            key = key_factory(cls) if key_factory else cls.__name__.lower()
            setattr(cls, key_attr, key)

        if not override and key in registry and not issubclass(cls, registry[key]):
            raise ValidationError(
                f"Class {cls!r} is already registered under key '{key}'. Rename the class or set a unique '{key_attr}'."
            )
        with lock:
            registry[key] = cls

        return cls  # type: ignore[return-value]

    if auto_register_serializer:
        serializer = make_registry_serializer(
            base_cls=base_cls, registry=registry, key_attr=key_attr
        )
        register_type_serializer(typ=base_cls, serializer=serializer)

    if auto_register_deserializer:
        deserializer = make_registry_deserializer(base_cls=base_cls, registry=registry)
        register_type_deserializer(typ=base_cls, deserializer=deserializer)

    registrar.__name__ = f"register_{base_cls.__name__.lower()}_type"
    return registrar


def make_registry_serializer(
    base_cls: type[SerializableT],
    registry: dict[str, type[SerializableT]],
    key_attr: str = "__type__",
) -> typing.Callable[[SerializableT], dict[str, typing.Any]]:
    """
    Creates a serializer function for a registry of `Serializable` subclasses.

    :param registry: The registry of `Serializable` subclasses.
    :param key_attr: The attribute used as the registry key.
    :return: A serializer function.
    """

    def serializer(obj: SerializableT) -> dict[str, typing.Any]:
        key = getattr(obj, key_attr, None)
        if not key or key not in registry:
            raise ValidationError(f"Unsupported {base_cls.__name__!r} type: {type(obj)!r}")

        return {key: obj.dump()}

    serializer.__name__ = f"serialize_{base_cls.__name__.lower()}"
    return serializer


def make_registry_deserializer(
    base_cls: type[SerializableT],
    registry: dict[str, type[SerializableT]],
) -> typing.Callable[[typing.Mapping[str, typing.Any]], SerializableT]:
    """
    Creates a deserializer function for a registry of `Serializable` subclasses.

    :param registry: The registry of `Serializable` subclasses.
    :param key_attr: The attribute used as the registry key.
    :return: A deserializer function.
    """

    def deserializer(data: typing.Mapping[str, typing.Any]) -> SerializableT:
        if not isinstance(data, Mapping) or len(data) != 1:
            raise DeserializationError("Invalid data format for deserialization.")

        key, value = next(iter(data.items()))
        if key not in registry:
            raise DeserializationError(f"Unsupported {base_cls.__name__!r} type: {key!r}")

        cls = registry[key]
        return cls.load(value)

    deserializer.__name__ = f"deserialize_{base_cls.__name__.lower()}"
    return deserializer
