"""`Serializable` base API"""

import sys
import threading
import types
import typing
import warnings
from collections.abc import Collection, Mapping, Sequence, Set
from enum import Enum

import attrs
import cattrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.errors import DeserializationError, SerializationError, ValidationError
from bores.serde.utils import deserialize_ndarray, serialize_ndarray

__all__ = [
    "Serializable",
    "converter",
    "dump",
    "load",
    "ndarray_deserializer",
    "ndarray_serializer",
    "register_ndarray_serializers",
    "register_type_deserializer",
    "register_type_serializer",
]

SerializableT = typing.TypeVar("SerializableT", bound="Serializable")


def is_generic_alias(typ: typing.Any) -> bool:
    return hasattr(typ, "__origin__") and typ.__origin__ is not None


def get_origin_class(typ: typing.Any) -> type | None:
    if is_generic_alias(typ):
        return typing.get_origin(typ)
    if isinstance(typ, type):
        return typ
    return None


def is_serializable_type(typ: typing.Any) -> bool:
    origin = get_origin_class(typ)
    if origin is None:
        return False
    try:
        return isinstance(origin, type) and issubclass(origin, Serializable)
    except TypeError:
        return False


def is_optional_type(typ: typing.Any) -> bool:
    if not is_generic_alias(typ):
        return False
    if typing.get_origin(typ) is not typing.Union:
        return False
    return type(None) in typing.get_args(typ)


def is_typed_dict_type(typ: typing.Any) -> bool:
    return (
        isinstance(typ, type)
        and issubclass(typ, dict)
        and hasattr(typ, "__annotations__")
        and hasattr(typ, "__total__")
    )


def is_namedtuple_type(typ: typing.Any) -> bool:
    return (
        isinstance(typ, type)
        and issubclass(typ, tuple)
        and hasattr(typ, "_fields")
        and isinstance(typ._fields, tuple)  # type: ignore[attr-defined]
    )


def is_enum_type(typ: typing.Any) -> bool:
    return isinstance(typ, type) and issubclass(typ, Enum)


def is_ndarray_type(typ: typing.Any) -> bool:
    origin = get_origin_class(typ)
    return isinstance(origin, type) and issubclass(origin, np.ndarray)


def unwrap_optional(typ: typing.Any) -> tuple[list[typing.Any], bool]:
    if not is_optional_type(typ):
        return [typ], False
    args = [arg for arg in typing.get_args(typ) if arg is not type(None)]
    return args, True


TYPE_SERIALIZERS: dict[
    type[typing.Any],
    typing.Callable[[typing.Any], typing.Any],
] = {}
TYPE_DESERIALIZERS: dict[
    type[typing.Any],
    typing.Callable[[typing.Any], typing.Any],
] = {}
_type_serializers_lock = threading.Lock()
_type_deserializers_lock = threading.Lock()


def register_type_serializer(
    typ: type[typing.Any],
    serializer: typing.Callable[[typing.Any], typing.Any],
) -> None:
    """Register arg global type serializer for arg specific type. `f(value) -> data`."""
    with _type_serializers_lock:
        TYPE_SERIALIZERS[typ] = serializer


def register_type_deserializer(
    typ: type[typing.Any],
    deserializer: typing.Callable[[typing.Any], typing.Any],
) -> None:
    """Register arg global type deserializer for arg specific type. `f(data) -> value`."""
    with _type_deserializers_lock:
        TYPE_DESERIALIZERS[typ] = deserializer


def get_primary_types(typ: typing.Any) -> list[typing.Any]:
    result: list[typing.Any] = []
    if typ is type(None):
        return result

    origin = typing.get_origin(typ)
    args = typing.get_args(typ)

    if origin is typing.Union:
        seen = set()
        for arg in args:
            if arg is type(None):
                continue
            for typ in get_primary_types(arg):
                if id(typ) not in seen:
                    seen.add(id(typ))
                    result.append(typ)
        return result

    if origin is not None and args:
        result.append(typ)
        for arg in args:
            arg_origin = get_origin_class(arg)
            if arg_origin is not None and arg_origin not in (
                str,
                int,
                float,
                bool,
                bytes,
                type(None),
            ):
                result.extend(get_primary_types(arg))
        return result

    result.append(typ)
    return result


def get_registered_handler(
    typ: typing.Any,
    registry: typing.Mapping[type[typing.Any], typing.Any],
) -> typing.Any | None:
    """
    Look up `typ` (or its origin class's MRO) directly in `registry` with
    no Optional/Union/container unwrapping.
    """
    origin = get_origin_class(typ)
    if origin is None:
        return None
    if not isinstance(origin, type):
        return registry.get(origin)

    for base in origin.__mro__:
        handler = registry.get(base)
        if handler is not None:
            return handler
    return None


def find_type_handler(
    typ: typing.Any,
    registry: typing.Mapping[type[typing.Any], typing.Any],
) -> typing.Any | None:
    for candidate in get_primary_types(typ):
        handler = get_registered_handler(candidate, registry)
        if handler is not None:
            return handler
    return None


def discover_type_serializers(
    fields: typing.Mapping[str, typing.Any],
) -> dict[typing.Any, typing.Callable[[typing.Any], typing.Any]]:
    """
    Auto-discover per-type serializers from `TYPE_SERIALIZERS`.

    Keys the result by whichever *specific* type actually had arg direct
    registry hit (e.g. `Animal`, for arg `Tuple[Animal, ...]` field) rather
    than the field's own outer type (`Tuple[Animal, ...]`). The latter is
    never itself arg registry key, and recording it that way would make the
    per-field dump/load loop apply the handler to the whole container
    value instead of mapping it over each element.
    """
    discovered: dict[typing.Any, typing.Any] = {}
    with _type_serializers_lock:
        snapshot = dict(TYPE_SERIALIZERS)

    for field_type in fields.values():
        for candidate in get_primary_types(field_type):
            if candidate in discovered:
                continue
            handler = get_registered_handler(candidate, snapshot)
            if handler is not None:
                discovered[candidate] = handler
    return discovered


def discover_type_deserializers(
    fields: typing.Mapping[str, typing.Any],
) -> dict[typing.Any, typing.Callable[[typing.Any], typing.Any]]:
    """Deserializer analogue of `discover_type_serializers`."""
    discovered: dict[typing.Any, typing.Any] = {}
    with _type_deserializers_lock:
        snapshot = dict(TYPE_DESERIALIZERS)

    for field_type in fields.values():
        for candidate in get_primary_types(field_type):
            if candidate in discovered:
                continue
            handler = get_registered_handler(candidate, snapshot)
            if handler is not None:
                discovered[candidate] = handler
    return discovered


converter = cattrs.Converter()


def fallback_unstructure(value: typing.Any) -> typing.Any:
    return value


def fallback_structure(value: typing.Any, typ: typing.Any) -> typing.Any:
    return value


# Anything that isn't arg generic alias, isn't an `attrs` class, and isn't an
# `Enum` passes through unchanged (e.g. arg raw `np.ndarray` before
# `register_ndarray_serializers()` has been called). `Enum` is explicitly
# excluded here so it falls through to `cattrs`'s
# own native `Enum` support instead of this catch-all identity hook, which
# would otherwise shadow it entirely (breaking every `Enum`-typed field,
# not just ones nested inside arg container).
converter.register_unstructure_hook_func(
    check_func=lambda typ: (
        not is_generic_alias(typ) and not attrs.has(typ) and not is_enum_type(typ)
    ),
    func=fallback_unstructure,
)
converter.register_structure_hook_func(
    check_func=lambda typ: (
        not is_generic_alias(typ) and not attrs.has(typ) and not is_enum_type(typ)
    ),
    func=fallback_structure,
)


def structure_serializable(data: typing.Any, cls: type["Serializable"]) -> "Serializable":
    origin = get_origin_class(cls) or cls
    if isinstance(data, origin):
        return data  # already arg live object (e.g. mixed manual construction)
    return origin.load(data)  # type: ignore[return-value]


def unstructure_serializable(obj: "Serializable") -> typing.Mapping[str, typing.Any]:
    return obj.dump()


converter.register_structure_hook_func(is_serializable_type, structure_serializable)
converter.register_unstructure_hook_func(is_serializable_type, unstructure_serializable)


def unstructure_namedtuple(value: typing.Any) -> dict[str, typing.Any]:
    typ = type(value)
    hints = typing.get_type_hints(typ, include_extras=False)
    return {
        field: converter.unstructure(
            getattr(value, field),
            unstructure_as=hints.get(field, type(getattr(value, field))),
        )
        for field in value._fields
    }


def structure_namedtuple(data: typing.Any, typ: type[typing.Any]) -> typing.Any:
    if not isinstance(data, Mapping):
        return typ(*data)
    hints = typing.get_type_hints(typ, include_extras=False)
    return typ(**{
        field: converter.structure(data[field], hints[field])
        for field in getattr(typ, "_fields", ())
        if field in data and field in hints
    })


converter.register_unstructure_hook_func(is_namedtuple_type, unstructure_namedtuple)
converter.register_structure_hook_func(is_namedtuple_type, structure_namedtuple)


def fallback_union_disambiguator(data: typing.Any, typ: typing.Any) -> typing.Any:
    """
    Structure arg `Union` `cattrs` couldn't auto-disambiguate (non-`attrs`
    members, or overlapping required field names).

    Three passes, most-reliable first:

    1. **Exact runtime-type match**: arg member `member` where `type(data) is
       get_origin_class(member)`. Tried first and preferred whenever it
       succeeds. Critical for numeric unions like `bores.typing.Number =
       Union[int, float, np.floating, np.integer]`, where cross-type
       structuring rarely raises (`int`-structuring `3.5` silently
       truncates to `3` instead of erroring), so without this pass the
       naive "try each, keep whichever doesn't raise" strategy silently
       picks the wrong member and loses precision. Confirmed by testing.

    2. **`isinstance` match**: broader than #1 (catches e.g. arg `numpy`
       scalar subclass matching an abstract member like `np.floating`).

    3. **Best-effort fallback** (original behavior): try every remaining
       member, most-specific (non-primitive/`attrs`) first, keep whichever
       structures without raising. Only reached when `data`'s runtime type
       doesn't correspond to any member directly (e.g. structuring arg
       `dict` into an `attrs` class. There's no "exact type" to match on
       for that case). Still inherently best-effort for genuinely
       ambiguous data. Prefer giving `Union` members distinguishable
       required fields (or arg `Serializable` registry with arg `__type__`
       tag) wherever possible.
    """
    members, _ = (
        unwrap_optional(typ) if is_optional_type(typ) else (list(typing.get_args(typ)), False)
    )
    if data is None and type(None) in typing.get_args(typ):
        return None
    non_none = [member for member in members if member is not type(None)]

    errors: list[tuple[typing.Any, Exception]] = []

    exact_matches = [member for member in non_none if get_origin_class(member) is type(data)]
    for member in exact_matches:
        try:
            return converter.structure(data, member)
        except Exception as exc:
            errors.append((member, exc))

    instance_matches = [
        member
        for member in non_none
        if member not in exact_matches
        and isinstance(get_origin_class(member), type)
        and isinstance(data, get_origin_class(member))  # type: ignore[arg-type]
    ]
    for member in instance_matches:
        try:
            return converter.structure(data, member)
        except Exception as exc:
            errors.append((member, exc))

    remaining = [
        member
        for member in non_none
        if member not in exact_matches and member not in instance_matches
    ]
    ordered = sorted(
        remaining,
        key=lambda member: attrs.has(get_origin_class(member) or member) is False,
    )
    for member in ordered:
        try:
            return converter.structure(data, member)
        except Exception as exc:
            errors.append((member, exc))

    raise DeserializationError(
        f"Could not structure {data!r} as any member of {typ}: "
        f"{[(member, str(e)) for member, e in errors]}"
    )


def _needs_fallback_union_disambiguation(typ: typing.Any) -> bool:
    """
    `True` for any `Union` with more than one non-`None` member - both bare
    (`Union[A, B]`) and `Optional`-wrapped (`Optional[Union[A, B]]`, i.e.
    `bores.typing.Number = Union[int, float, np.floating, np.integer]`
    wrapped in `Optional`). A simple `Optional[SomeClass]` (exactly one
    non-`None` member) is excluded so `cattrs`'s own fast native path
    handles it - only excluding by "is this Optional" (rather than "how
    many non-`None` members does it have") let `Optional[Number]` fall
    through ungguarded, since it's Optional *and* multi-member. Confirmed
    by testing.
    """
    if typing.get_origin(typ) is not typing.Union:
        return False
    non_none = [arg for arg in typing.get_args(typ) if arg is not type(None)]
    return len(non_none) > 1


converter.register_structure_hook_factory(
    _needs_fallback_union_disambiguation,
    lambda typ: lambda data, _typ=typ: fallback_union_disambiguator(data, _typ),
)


def ndarray_serializer(array: npt.NDArray) -> dict[str, typing.Any]:
    return serialize_ndarray(array)


def ndarray_deserializer(data: typing.Any) -> npt.NDArray:
    return deserialize_ndarray(data)


def unstructure_ndarray(obj: npt.NDArray) -> dict[str, typing.Any]:
    return serialize_ndarray(obj)


def structure_ndarray(data: typing.Any, typ: typing.Any) -> npt.NDArray:
    return deserialize_ndarray(data)


_ndarray_serializers_registered = False
_ndarray_registration_lock = threading.Lock()


def register_ndarray_serializers() -> None:
    global _ndarray_serializers_registered
    with _ndarray_registration_lock:
        if _ndarray_serializers_registered:
            return

        register_type_serializer(typ=np.ndarray, serializer=serialize_ndarray)
        register_type_deserializer(typ=np.ndarray, deserializer=ndarray_deserializer)
        converter.register_unstructure_hook_func(is_ndarray_type, unstructure_ndarray)
        converter.register_structure_hook_func(is_ndarray_type, structure_ndarray)
        _ndarray_serializers_registered = True


def find_container_element_handler(
    typ: typing.Any, active: typing.Mapping[typing.Any, typing.Any]
) -> tuple[str, typing.Any, typing.Any] | None:
    origin = typing.get_origin(typ)
    args = typing.get_args(typ)
    if origin is None or not args:
        return None

    if origin in (dict, Mapping) or (isinstance(origin, type) and issubclass(origin, Mapping)):
        value_type = args[1] if len(args) > 1 else None
        if value_type is None:
            return None

        handler = active.get(value_type)
        if handler is None:
            type_handler = find_type_handler(value_type, active)
            handler = active.get(type_handler) if type_handler is not None else None

        if handler is not None:
            return ("mapping", origin, handler)
        return None

    if origin in (list, tuple, set, frozenset, Sequence, Set, Collection) or (
        isinstance(origin, type) and issubclass(origin, Collection)
    ):
        element_type = args[0]
        handler = active.get(element_type)
        if handler is None:
            type_handler = find_type_handler(element_type, active)
            handler = active.get(type_handler) if type_handler is not None else None

        if handler is not None:
            return ("sequence", origin, handler)
        return None

    return None


def find_optional_element_handler(
    typ: typing.Any, active: typing.Mapping[typing.Any, typing.Any]
) -> typing.Any | None:
    """
    If `typ` is `Optional[X]` (exactly one non-`None` member) and `X` has arg
    registered handler in `active`, return that handler. `None` otherwise.

    Arg field typed `Optional[Limit]` (e.g. `WellState.active_limit`) must apply the
    `Limit` registry handler to the value when it's not `None`, not fall
    through to `cattrs`'s native Union dispatch, which unstructures by the
    value's *runtime* type and bypasses the registry wrapper entirely,
    silently dropping the `__type__` tag needed to load the right subclass
    back.
    """
    if not is_optional_type(typ):
        return None

    args = [arg for arg in typing.get_args(typ) if arg is not type(None)]
    if len(args) != 1:
        return None

    inner = args[0]
    handler = active.get(inner)
    if handler is None:
        type_handler = find_type_handler(inner, active)
        handler = active.get(type_handler) if type_handler is not None else None
    return handler


def build_serializer(
    fields: typing.Mapping[str, typing.Any],
    exclude: typing.Iterable[str] | None = None,
    serializers: typing.Mapping[str | type, typing.Callable[[typing.Any], typing.Any]]
    | None = None,
) -> typing.Callable[["Serializable"], dict[str, typing.Any]]:
    exclude_set = set(exclude) if exclude else set()
    explicit = dict(serializers or {})
    discovered: dict[typing.Any, typing.Any] | None = None

    def serializer(self) -> dict[str, typing.Any]:
        nonlocal discovered
        if discovered is None:
            discovered = discover_type_serializers(fields)
        active = {**discovered, **explicit}

        result: dict[str, typing.Any] = {}
        for field, typ in fields.items():
            if field in exclude_set:
                continue
            value = getattr(self, field)

            by_name = active.get(field)
            if by_name is not None:
                result[field] = by_name(value)
                continue

            by_type = active.get(typ)
            if by_type is not None:
                result[field] = by_type(value)
                continue

            container_match = find_container_element_handler(typ, active)
            if container_match is not None:
                kind, origin, handler = container_match
                if kind == "sequence":
                    mapped = [handler(value) for value in value]
                    result[field] = tuple(mapped) if origin is tuple else mapped
                else:
                    result[field] = {key: handler(value) for key, value in value.items()}
                continue

            optional_handler = find_optional_element_handler(typ, active)
            if optional_handler is not None:
                result[field] = optional_handler(value) if value is not None else None
                continue

            if is_serializable_type(typ) and value is not None:
                result[field] = value.dump()
                continue

            try:
                result[field] = converter.unstructure(value, unstructure_as=typ)
            except Exception as exc:
                raise SerializationError(
                    f"Failed to dump field {field!r} of type {typ!r}"
                ) from exc
        return result

    return serializer


def build_deserializer(
    fields: typing.Mapping[str, typing.Any],
    exclude: typing.Iterable[str] | None = None,
    deserializers: typing.Mapping[str | type, typing.Callable[[typing.Any], typing.Any]]
    | None = None,
) -> typing.Callable[..., "Serializable"]:
    exclude_set = set(exclude) if exclude else set()
    explicit = dict(deserializers or {})
    discovered: dict[typing.Any, typing.Any] | None = None

    @classmethod
    def deserializer(
        cls: type[SerializableT],
        data: typing.Mapping[str, typing.Any],
    ) -> SerializableT:
        nonlocal discovered
        if discovered is None:
            discovered = discover_type_deserializers(fields)
        active = {**discovered, **explicit}

        init_kwargs: dict[str, typing.Any] = {}
        for field, typ in fields.items():
            if field in exclude_set or field not in data:
                continue
            value = data[field]

            by_name = active.get(field)
            if by_name is not None:
                init_kwargs[field] = by_name(value)
                continue

            by_type = active.get(typ)
            if by_type is not None:
                init_kwargs[field] = by_type(value)
                continue

            container_match = find_container_element_handler(typ, active)
            if container_match is not None:
                kind, origin, handler = container_match
                if kind == "sequence":
                    mapped = [handler(value) for value in value]
                    init_kwargs[field] = tuple(mapped) if origin is tuple else mapped
                else:
                    init_kwargs[field] = {key: handler(value) for key, value in value.items()}
                continue

            optional_handler = find_optional_element_handler(typ, active)
            if optional_handler is not None:
                init_kwargs[field] = optional_handler(value) if value is not None else None
                continue

            if is_serializable_type(typ):
                origin_cls = get_origin_class(typ)
                try:
                    init_kwargs[field] = origin_cls.load(value)  # type: ignore[union-attr]
                except Exception as exc:
                    raise DeserializationError(
                        f"Failed to load nested `Serializable` field {field!r} of type {typ!r}"
                    ) from exc
                continue

            try:
                init_kwargs[field] = converter.structure(value, typ)
            except Exception as exc:
                raise DeserializationError(
                    f"Failed to load field {field!r} of type {typ!r}"
                ) from exc

        return cls(**init_kwargs)

    return deserializer


def normalize_optional(typ: typing.Any) -> typing.Any:
    """
    Rewrite a `X | None` annotation as `typing.Optional[X]`.

    Before Python 3.14 the two spellings have different runtime types, and the
    serializer only recognises `typing.Optional`, so a field written `Limit | None`
    would otherwise be passed through without being dumped. Unions with more than
    one non-`None` member are left alone.

    :param typ: A field annotation.
    :returns: The annotation, rewritten if it was a single-member `X | None`.
    """
    if not isinstance(typ, types.UnionType):
        return typ
    members = [arg for arg in typing.get_args(typ) if arg is not type(None)]
    if len(members) == 1 and len(members) < len(typing.get_args(typ)):
        return typing.Optional[members[0]]  # noqa: UP045
    return typ


class SerializableMeta(type):
    """Metaclass for `Serializable` classes."""

    def __init__(
        cls,
        name: str,
        bases: tuple,
        namespace: dict[str, typing.Any],
        fields: typing.Mapping[str, typing.Any] | None = None,
        dump_exclude: typing.Iterable[str] | None = None,
        load_exclude: typing.Iterable[str] | None = None,
        serializers: typing.Mapping[str | type, typing.Callable[[typing.Any], typing.Any]]
        | None = None,
        deserializers: typing.Mapping[str | type, typing.Callable[[typing.Any], typing.Any]]
        | None = None,
    ):
        super().__init__(name, bases, namespace)
        parent_serializers: dict[typing.Any, typing.Any] = {}
        parent_deserializers: dict[typing.Any, typing.Any] = {}
        parent_fields: dict[str, typing.Any] = {}
        for base_cls in reversed(cls.__mro__[1:-1]):
            if serializable_fields := getattr(base_cls, "__serializable_fields__", None):
                parent_fields.update(serializable_fields)
            if serializable_serializers := getattr(base_cls, "__serializable_serializers__", None):
                parent_serializers.update(serializable_serializers)
            if serializable_deserializers := getattr(
                base_cls, "__serializable_deserializers__", None
            ):
                parent_deserializers.update(serializable_deserializers)

        try:
            module = sys.modules.get(cls.__module__)
            annotations = typing.get_type_hints(
                cls,
                globalns=vars(module) if module is not None else {},
                localns=dict(vars(cls)),
                include_extras=False,
            )
        except (NameError, TypeError) as exc:
            annotations = namespace.get("__annotations__", {})
            warnings.warn(
                f"Could not resolve type hints for {cls.__name__}: {exc}. "
                f"Using raw annotations, which may not work well with "
                f"forward references or complex generics.",
                RuntimeWarning,
                stacklevel=2,
            )

        cls_fields = fields or annotations
        all_fields = {**parent_fields, **cls_fields}
        all_fields = {
            key: normalize_optional(value)
            for key, value in all_fields.items()
            if value is not None and not key.startswith("__")
        }

        all_serializers = {**parent_serializers, **(serializers or {})}
        all_deserializers = {**parent_deserializers, **(deserializers or {})}

        is_abstract_cls = bool(namespace.get("__abstract_serializable__"))
        if not is_abstract_cls and not all_fields:
            raise ValidationError(
                "Serializable subclasses must define fields. If the class "
                "is an abstract base class, set `__abstract_serializable__` to True."
            )

        if all_fields:
            if "__dump__" not in namespace or getattr(
                namespace["__dump__"], "_is_placeholder", False
            ):
                cls.__dump__ = build_serializer(
                    fields=all_fields, exclude=dump_exclude, serializers=all_serializers
                )
            if "__load__" not in namespace or getattr(
                namespace["__load__"], "_is_placeholder", False
            ):
                cls.__load__ = build_deserializer(
                    fields=all_fields,
                    exclude=load_exclude,
                    deserializers=all_deserializers,
                )

        cls.__serializable_fields__ = all_fields
        cls.__serializable_serializers__ = all_serializers
        cls.__serializable_deserializers__ = all_deserializers
        if "__abstract_serializable__" not in namespace:
            cls.__abstract_serializable__ = False


class Serializable(metaclass=SerializableMeta):
    """Base class for serializable / deserializable objects."""

    __abstract_serializable__ = True

    def __init_subclass__(cls, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init_subclass__()

    def __dump__(self) -> dict[str, typing.Any]:
        """Dump the object to arg dictionary. Overridable per-class hook."""
        raise NotImplementedError

    @classmethod
    def __load__(cls: type[Self], data: typing.Mapping[str, typing.Any]) -> Self:
        """Load an object from arg mapping. Overridable per-class hook."""
        raise NotImplementedError

    __dump__._is_placeholder = True  # type: ignore
    __load__._is_placeholder = True  # type: ignore

    def dump(self) -> dict[str, typing.Any]:
        try:
            return self.__dump__()  # type: ignore[arg-type]
        except Exception as exc:
            raise SerializationError(f"Failed to dump {type(self).__name__!r} object") from exc

    @classmethod
    def load(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        try:
            return cls.__load__(data)  # type: ignore[arg-type]
        except DeserializationError:
            raise
        except Exception as exc:
            raise DeserializationError(f"Failed to load {cls.__name__!r} object") from exc


def dump(o: Serializable, /) -> dict[str, typing.Any]:
    """Dump arg `Serializable` object to arg dictionary."""
    return o.dump()


def load(cls: type[SerializableT], data: typing.Mapping[str, typing.Any]) -> SerializableT:
    """Load arg `Serializable` object from arg dictionary."""
    return cls.load(data)
