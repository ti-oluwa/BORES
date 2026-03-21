"""State storage backends."""

import base64
import functools
import logging
import shutil
import sys
import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import version
from os import PathLike
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import orjson
import yaml
import zarr  # type: ignore[import-untyped]
from packaging.version import Version
from typing_extensions import ParamSpec, Self
from zarr.storage import StoreLike  # type: ignore[import-untyped]

from bores.errors import StorageError, ValidationError
from bores.serialization import Serializable, SerializableT
from bores.utils import safe_json_dumps, safe_json_loads

__all__ = [
    "HDF5Store",
    "JSONStore",
    "YAMLStore",
    "ZarrStore",
    "data_store",
    "new_store",
]

IS_PYTHON_310_OR_LOWER = sys.version_info < (3, 11)
ZARR_VERSION_GTE_3 = Version(version("zarr")) >= Version("3.0.0")


logger = logging.getLogger(__name__)


DataValidator = typing.Callable[[SerializableT], SerializableT]
HandleT = typing.TypeVar("HandleT")


class EntryMeta(typing.NamedTuple):
    """
    Lightweight record describing one persisted item.

    Stored alongside each entry so the store can answer index-based and
    predicate-based queries without deserialising any payload data.
    """

    idx: int
    """Zero-based position in insertion order."""
    group_name: str
    """Internal storage key (opaque to callers)."""
    meta: typing.Dict[str, str]
    """JSON serializable metadata dictionary"""


class DataStore(ABC, typing.Generic[SerializableT, HandleT]):
    """
    Abstract base class for all storage backends.

    Every backend maintains a compact metadata index (`list[EntryMeta]`) so
    callers can inspect stored entries and jump directly to specific ones without
    a full scan.  Group naming is internal and fixed — callers never supply it.
    All writes overwrite existing content.

    **Persistent handle (open / close / __call__)**

    By default every method (`dump`, `load`, `append`, `entries`) opens
    the underlying file/directory, performs its work, and closes it again.  For
    workloads that call `append` in a tight loop (e.g. a background I/O
    thread) this per-call overhead is significant.

    Call `open(**kwargs)` once to obtain a persistent handle that all
    subsequent methods will reuse.  Call `close()` when finished.  The
    `__call__(**kwargs)` context manager does both automatically:

    ```python
    # Low-level
    store.open(mode="a")
    for state in states:
        store.append(state)
    store.close()

    # Context manager (Preferred)
    with store(mode="a"):
        for state in states:
            store.append(state)
    ```

    When no handle is open (`_handle is None`) every method falls back to
    opening and closing internally, so existing call-sites require no changes.

    **Interface**
    `dump(data)`
        Persist an iterable of `Serializable` items.  Always overwrites.

    `load(typ)`
        Load every item.  Returns a generator.

    `load(typ, indices=[0, 3, 7])`
        Load only the items at the given positional indices.

    `load(typ, predicate=lambda e: e.idx < 10)`
        Load only items whose `EntryMeta` satisfies *predicate*.

    `entries()`
        Return the full `list[EntryMeta]` without deserialising any payload.
        Uses this for `count()`, `max_index()`, membership checks, etc.
    """

    supports_append: bool = False

    def __init__(self) -> None:
        self._handle: typing.Optional[HandleT] = None
        """The open handle. `None` means "no persistent handle; open/close per call"."""

    @abstractmethod
    def open(self, **kwargs: typing.Any) -> None:
        """
        Open the underlying storage and attach the handle to `self._handle`.

        Subsequent calls to `dump`, `load`, `append`, and `entries`
        will use this handle instead of opening the file themselves.

        :param kwargs: Backend-specific keyword arguments (e.g. `mode="a"`).
        :raises StorageError: If the store cannot be opened.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Flush and release the persistent handle (`self._handle`).

        After this call `self._handle` must be `None`.  It is safe to call
        `close()` when no handle is open.

        Implementations should be idempotent.
        """
        ...

    @contextmanager
    def __call__(self, **kwargs: typing.Any) -> typing.Generator[Self, None, None]:
        """
        Context manager that opens the store, yields `self`, then closes it.

        Usage:

        ```python
        with store(mode="a") as s:
            for item in items:
                s.append(item)
        ```

        :param kwargs: Forwarded verbatim to `open(**kwargs)`.
        :raises StorageError: Re-raised from `open` or `close`.
        """
        self.open(**kwargs)
        try:
            yield self
        finally:
            self.close()

    @abstractmethod
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        """
        Persist *data*, always overwriting any existing content in the store.

        Every item in *data* is written in iteration order.  If the backing
        file or directory already exists it is truncated first, so calling
        `dump` twice is equivalent to calling it once with the second dataset.
        Use `append` when you need to add items to an existing store without
        discarding what is already there.

        :param data: Iterable of `Serializable` instances to persist.
        :param validator: Optional callable applied to each item before it is
            written.  Receives the item and must return a (possibly transformed)
            item of the same type.  Raise to abort persistence of that item.
        :param meta: Optional callable that receives each item and returns a
            plain `dict` of JSON-serialisable values (str, int, float, bool).
            The returned dict is stored alongside the entry and surfaced on
            `EntryMeta.meta`, making it available for zero-deserialisation
            filtering in `load` and `entries`.
        """
        ...

    @abstractmethod
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load and yield items from the store in insertion order.

        Filtering is applied before any array data is deserialised, so entries
        that do not match have no I/O cost beyond reading their metadata.
        When both `indices` and `predicate` are supplied, `indices` takes
        priority and `predicate` is ignored.

        :param typ: The `Serializable` subclass to deserialise each entry into.
        :param indices: If given, load only the entries at these zero-based
            insertion-order positions.  Out-of-range indices raise `IndexError`.
        :param predicate: If given (and `indices` is `None`), yield only entries
            for which `predicate(entry_meta)` returns `True`.  The predicate
            receives an `EntryMeta` instance and may inspect `entry_meta.meta`
            to filter on stored metadata without touching array data.
        :param validator: Optional callable applied to each deserialised item
            before it is yielded.  Receives the item and must return a
            (possibly transformed) item of the same type.
        :return: Generator yielding deserialised items matching the filter.
        """
        ...

    def append(
        self,
        item: SerializableT,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        """
        Append a single item to the store without rewriting existing entries.

        The item is assigned the next available insertion-order index.  Unlike
        `dump`, existing entries are never touched.  Backends that do not
        support append-style writes raise `NotImplementedError` at call time
        rather than at construction time.

        Check `supports_append` before calling if in doubt.

        :param item: The `Serializable` instance to persist.
        :param validator: Optional callable applied to *item* before it is
            written.  Receives the item and must return a (possibly transformed)
            item of the same type.  Raise to abort the write.
        :param meta: Optional callable that receives *item* and returns a plain
            `dict` of JSON-serialisable values stored on `EntryMeta.meta`.
            Use this to record lightweight metadata (e.g. `{"step": state.step}`)
            that can later be used to filter entries via `load(predicate=...)`
            or `entries()` without deserialising array data.
        :return: The `EntryMeta` record created for the appended item, including
            its assigned index, group name, and any stored metadata.
        :raises NotImplementedError: If the backend does not support appending.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__!r} does not implement `append(...)`"
        )

    @abstractmethod
    def entries(self) -> typing.List[EntryMeta]:
        """
        Return metadata for every stored item in insertion order.

        This method must not deserialise any payload data.  Implementations
        should read only group names, file keys, and lightweight attributes,
        never array datasets.  The returned list can therefore be used for
        cheap introspection (counts, step lookups, predicate filtering) without
        triggering any significant I/O.

        :return: List of `EntryMeta` instances in insertion order.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush the data store clean. Clear every data item stored."""
        ...

    def count(self) -> int:
        """
        Return the number of items currently stored.

        Delegates to `entries()` and returns its length.  No payload data is
        deserialised.

        :return: Total number of stored entries.
        """
        return len(self.entries())

    def max_index(self) -> typing.Optional[int]:
        """
        Return the highest insertion-order index in the store, or `None` if empty.

        Useful for targeting the last written entry via
        `store.load(typ, indices=[store.max_index()])` without replaying
        the entire store.  No payload data is deserialised.

        :return: The highest `EntryMeta.index` value, or `None` if the store
            contains no entries.
        """
        metas = self.entries()
        return max(e.idx for e in metas) if metas else None


StoreT = typing.TypeVar("StoreT", bound=DataStore)

_STORAGE_BACKENDS: typing.Dict[str, typing.Type[DataStore]] = {}


@typing.overload
def data_store(
    *names: str,
    store_cls: typing.Type[StoreT],
) -> typing.Type[StoreT]: ...


@typing.overload
def data_store(
    *names: str,
) -> typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]: ...


def data_store(
    *names: str,
    store_cls: typing.Optional[typing.Type[StoreT]] = None,
) -> typing.Union[
    typing.Type[StoreT], typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]
]:
    """
    Data store registration decorator.

    Register a data store class with a given name.

    :param name: Name of the data store backend
    :param store_cls: Data store class to register
    :return: Decorator function if `store_cls` is None, else None
    """

    def _decorator(store_cls: typing.Type[StoreT]) -> typing.Type[StoreT]:
        for name in names:
            _STORAGE_BACKENDS[name] = store_cls
        return store_cls

    if store_cls is not None:
        return _decorator(store_cls)
    return _decorator


def _validate_filepath(
    filepath: typing.Union[PathLike, str],
    expected_extension: typing.Optional[str] = None,
    is_directory: bool = False,
    create_if_not_exists: bool = False,
) -> Path:
    """
    Validate and normalize a filepath for state storage.

    :param filepath: Path to validate
    :param expected_extension: Expected file extension (e.g., '.pkl', '.h5', '.npz')
        If None, no extension validation is performed
    :param is_directory: If True, validates that the path is suitable for a directory
        (no extension or matches expected extension for directory-based stores)
    :param create_if_not_exists: If True, creates the file/directory if it does not exist
    :return: Validated Path object
    :raises StorageError: If filepath is invalid or has wrong extension
    """
    path = Path(filepath)

    if not str(path).strip():
        raise StorageError("Filepath cannot be empty")

    if "\x00" in str(path):
        raise StorageError("Filepath contains null characters")

    if is_directory:
        if expected_extension and path.suffix:
            if expected_extension not in path.suffixes:
                raise StorageError(
                    f"Directory-based store expected extension '{expected_extension}', "
                    f"got '{''.join(path.suffixes)}'. Use '{path.with_suffix(expected_extension)}' instead."
                )
        elif path.suffix and ".zarr" not in path.suffixes:
            logger.warning(
                f"Path '{path}' has extension '{path.suffix}' but will be treated as a directory. "
                f"Consider using a name without extension or '.zarr' for clarity."
            )
        return path

    if expected_extension:
        if not path.suffix:
            path = path.with_suffix(expected_extension)
            logger.debug(f"Added extension: {path}")
        elif expected_extension not in path.suffixes:
            raise StorageError(
                f"Expected file extension '{expected_extension}', got '{''.join(path.suffixes)}'. "
                f"Use '{path.with_suffix(expected_extension)}' instead."
            )

    if create_if_not_exists:
        is_file = path.suffix != ""
        directory = path.parent if is_file else path
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created parent directory: {directory}")
            except Exception as exc:
                raise StorageError(
                    f"Failed to create parent directory '{directory}': {exc}"
                ) from exc

        if is_file and not path.exists():
            try:
                path.touch(exist_ok=True)
                logger.debug(f"Created file: {path}")
            except Exception as exc:
                raise StorageError(f"Failed to create file '{path}': {exc}") from exc
    return path


P = ParamSpec("P")
R = typing.TypeVar("R")


def reraise_storage_error(func: typing.Callable[P, R]) -> typing.Callable[P, R]:
    """
    Wraps a function to raise `StorageError` on exceptions.

    :param func: Function to wrap
    """

    @functools.wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(exc) from exc

    return _wrapper


# Sentinel for None values to avoid object dtype issues
_NONE_SENTINEL = "__NONE_SENTINEL__"


def _is_none_sentinel(value: typing.Any) -> bool:
    """Check if value is the None sentinel."""
    return isinstance(value, str) and value == _NONE_SENTINEL


def _sequence_to_ndarray(value: Sequence, path: str) -> np.typing.NDArray:
    """
    Convert a (possibly nested) sequence into a NumPy array
    that is safe for HDF5/Zarr storage.

    Raises if the sequence would produce dtype=object.
    """
    if not value:
        return np.empty((0,), dtype=np.int8)

    if any(isinstance(v, Mapping) for v in value):
        raise TypeError(
            f"Sequence of mappings must be stored as groups, not datasets: {path}"
        )

    if isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes)):
        arrays = [_sequence_to_ndarray(v, path) for v in value]
        try:
            arr = np.stack(arrays)
        except ValueError as exc:
            raise TypeError(f"Inconsistent nested sequence shapes at {path}") from exc
        return arr

    if all(isinstance(v, (bool, np.bool_)) for v in value):
        return np.asarray(value, dtype=bool)

    if all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        for v in value
    ):
        return np.asarray(value)

    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
        return np.asarray(value)

    if all(isinstance(v, str) for v in value):
        if hasattr(h5py, "string_dtype"):
            dtype = h5py.string_dtype(encoding="utf-8")
            return np.asarray(value, dtype=dtype)
        return np.asarray(value, dtype="U")

    raise TypeError(
        f"Unsupported or mixed sequence contents at {path}: "
        f"{set(type(v).__name__ for v in value)}"
    )


def _normalize_for_storage(value: typing.Any) -> typing.Any:
    """Normalize Python values for storage (replace None with sentinel)."""
    if value is None:
        return _NONE_SENTINEL
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, np.ndarray)
    ):
        return [_normalize_for_storage(v) for v in value]
    elif isinstance(value, Mapping):
        return {k: _normalize_for_storage(v) for k, v in value.items()}
    return value


def _denormalize_from_storage(value: typing.Any) -> typing.Any:
    """Denormalize values from storage (replace sentinel with None)."""
    if _is_none_sentinel(value):
        return None
    elif isinstance(value, list):
        return [_denormalize_from_storage(v) for v in value]
    elif isinstance(value, dict):
        return {k: _denormalize_from_storage(v) for k, v in value.items()}
    return value


def _normalize_loaded_value(value: typing.Any) -> typing.Any:
    """Normalize values loaded from HDF5/Zarr datasets."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S", "O"):
            result = value.astype(str).tolist()
            return _denormalize_from_storage(result)
        return value
    return _denormalize_from_storage(value)


def _normalize_loaded_mapping_sequence(value: typing.Any) -> typing.Any:
    """
    Detect if a loaded mapping represents a sequence and convert it back.

    Groups with numeric string keys "0", "1", "2", ... are sequences.
    """
    if (
        isinstance(value, dict)
        and value
        and all(isinstance(k, str) and k.isdigit() for k in value)
    ):
        max_idx = max(int(k) for k in value)
        result = []
        for i in range(max_idx + 1):
            key = str(i)
            if key in value:
                result.append(value[key])
            else:
                result.append(None)
        return result
    return value


def _get_group_name(index: int) -> str:
    """
    Returns a group name using fixed naming scheme: `entry_{index:010d}`.

    Zero-padded to 10 digits so lexicographic order == insertion order,
    meaning `sorted(keys)` always gives the correct traversal order.
    """
    return f"entry_{index:010d}"


def _get_index_from_group_name(name: str) -> typing.Optional[int]:
    """Parse group name of form `entry_NNNNNNNNNN` to integer index, or `None` if not our format."""
    if name.startswith("entry_") and len(name) == 16:
        try:
            return int(name[6:])
        except ValueError:
            return None
    return None


"""
DataStore with flattened entry layout.

**Encoding contract**
Path segments are percent-encoded so that the separator character (``→``,
U+2192) and the escape character (``%``) can never appear unescaped in a
segment.  This makes path encoding injective: two distinct nested paths can
never produce the same flat key.

    encode: "%" → "%25",  "→" → "%E2%86%92"
    decode: reverse of above

**Special value types**
The following non-array leaf types need extra round-trip help and are tagged
in a ``_vtypes`` attr dict stored once per entry group:

    "json" - list/dict that is not a numpy array (serialised via orjson)
    "none" - Python None  (stored as sentinel string, tagged for safety)
    "bool" - Python bool  (JSON round-trips fine but we tag for clarity)

Untagged scalars are int, float, or str and survive attrs round-trips natively.

**Empty sequences**
Written as a zero-length int8 dataset (same as the nested layout) so the
array-vs-scalar distinction is preserved.

**Collision detection**
A debug-mode assertion checks that no two source paths produce the same
encoded flat key.  This is a safeguard; in practice attrs field names cannot
collide after encoding.
"""


_SEP = "\u2192"  # → U+2192  RIGHTWARDS ARROW — path segment separator
_ESC = "%"  # percent   — escape character

_SEP_ENCODED = "%E2%86%92"
_ESC_ENCODED = "%25"


def _encode_segment(s: str) -> str:
    """Percent-encode `%` and `→` so neither can appear raw in a segment."""
    return s.replace(_ESC, _ESC_ENCODED).replace(_SEP, _SEP_ENCODED)


def _decode_segment(s: str) -> str:
    """Reverse of `_encode_segment`."""
    return s.replace(_SEP_ENCODED, _SEP).replace(_ESC_ENCODED, _ESC)


def _join_path(*segments: str) -> str:
    return _SEP.join(_encode_segment(s) for s in segments)


def _split_path(flat_key: str) -> typing.List[str]:
    return [_decode_segment(s) for s in flat_key.split(_SEP)]


_VTYPE_JSON = "json"  # list-of-mappings or arbitrary list/dict → orjson
_VTYPE_NONE = "none"  # Python None
_VTYPE_BOOL = "bool"  # Python bool

_INTERNAL = {"_vtypes", "_meta", "_index", "_group_name", "count", "version"}


def _flatten(
    data: typing.Mapping[str, typing.Any],
    prefix: typing.Tuple[str, ...] = (),
    out_arrays: typing.Optional[typing.Dict[str, np.typing.NDArray]] = None,
    out_scalars: typing.Optional[typing.Dict[str, typing.Any]] = None,
    out_vtypes: typing.Optional[typing.Dict[str, str]] = None,
) -> typing.Tuple[
    typing.Dict[str, np.ndarray],
    typing.Dict[str, typing.Any],
    typing.Dict[str, str],
]:
    """
    Recursively flatten *data* into three parallel flat dicts.

    :param data: Nested mapping to flatten.
    :param prefix: Current path prefix as a tuple of raw (unencoded) segments.
    :param out_arrays: Accumulator for `{flat_key: ndarray}` pairs.
    :param out_scalars: Accumulator for `{flat_key: scalar}` pairs.
    :param out_vtypes: Accumulator for `{flat_key: vtype_tag}` pairs
        (only entries that need a tag are included).
    :returns: `(arrays, scalars, vtypes)` flat dicts.
    """
    if out_arrays is None:
        out_arrays = {}
    if out_scalars is None:
        out_scalars = {}
    if out_vtypes is None:
        out_vtypes = {}

    for key, value in data.items():
        path = prefix + (key,)
        flat_key = _join_path(*path)

        if value is None:
            out_scalars[flat_key] = _NONE_SENTINEL
            out_vtypes[flat_key] = _VTYPE_NONE
            continue

        if isinstance(value, Mapping):
            _flatten(value, path, out_arrays, out_scalars, out_vtypes)
            continue

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise TypeError(f"Cannot store object-dtype array at path {flat_key!r}")
            out_arrays[flat_key] = value
            continue

        # Convert numpy scalars to Python native ---
        if isinstance(value, (np.integer, np.floating)):
            out_scalars[flat_key] = value.item()
            continue

        if isinstance(value, np.bool_):
            out_scalars[flat_key] = bool(value)
            out_vtypes[flat_key] = _VTYPE_BOOL
            continue

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                # Empty sequence → zero-length dataset
                out_arrays[flat_key] = np.empty((0,), dtype=np.int8)
                continue

            if isinstance(value[0], Mapping):
                # List of mappings → serialise as JSON scalar; not worth
                # the complexity of flattening further since these are always
                # small (well perforations, schedule entries, etc.)
                out_scalars[flat_key] = orjson.dumps(
                    [dict(item) for item in value], option=orjson.OPT_SERIALIZE_NUMPY
                ).decode()
                out_vtypes[flat_key] = _VTYPE_JSON
                continue

            # Homogeneous sequence of scalars/arrays → convert to ndarray
            arr = _sequence_to_ndarray(value, path=flat_key)
            out_arrays[flat_key] = arr
            continue

        if isinstance(value, bool):
            out_scalars[flat_key] = value
            out_vtypes[flat_key] = _VTYPE_BOOL
            continue

        # other scalars, int, float, str, etc.
        out_scalars[flat_key] = value

    return out_arrays, out_scalars, out_vtypes


def _unflatten(
    arrays: typing.Dict[str, np.typing.NDArray],
    scalars: typing.Dict[str, typing.Any],
    vtypes: typing.Dict[str, str],
) -> typing.Dict[str, typing.Any]:
    """
    Reconstruct a nested dict from the three flat dicts produced by `_flatten`.

    :param arrays: `{flat_key: ndarray}` — from zarr array datasets.
    :param scalars: `{flat_key: scalar}` — from zarr group attrs.
    :param vtypes: `{flat_key: vtype_tag}` — from the `_vtypes` attr.
    :returns: Reconstructed nested dict.
    """
    result: typing.Dict[str, typing.Any] = {}

    def _set_nested(d: typing.Dict, parts: typing.List[str], value: typing.Any) -> None:
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            elif not isinstance(d[part], dict):
                # A scalar was registered at a prefix that is also a path
                # prefix for deeper keys. This should never happen with
                # well-formed attrs classes, but we should guard anyway.
                raise StorageError(
                    f"Path conflict at segment {part!r}: "
                    f"expected dict, got {type(d[part]).__name__}"
                )
            d = d[part]
        d[parts[-1]] = value

    for flat_key, arr in arrays.items():
        parts = _split_path(flat_key)
        value = _normalize_loaded_value(arr)
        _set_nested(result, parts, value)

    for flat_key, raw in scalars.items():
        parts = _split_path(flat_key)
        vtype = vtypes.get(flat_key)

        if vtype == _VTYPE_NONE:
            value = None
        elif vtype == _VTYPE_JSON:
            value = orjson.loads(raw)
        elif vtype == _VTYPE_BOOL:
            value = bool(raw)
        else:
            value = _denormalize_from_storage(raw)

        _set_nested(result, parts, value)

    return result


@data_store("zarr")
class ZarrStore(DataStore[SerializableT, zarr.Group]):
    """
    Zarr-based storage.

    Fast, efficient compression with lazy loading.
    Good for large 3D numpy arrays.

    **Layout**

    ```mermaid
    <root.zarr>/
        entry_0000000000/          ← one group per item
            <encoded→path>         ← zarr dataset  (numpy arrays)
            attrs:
                <encoded→path>     ← scalar values
                _scalars_encoded→path: value
                _vtypes:           ← type tags for non-trivial scalars
                _meta:             ← user metadata
                _index:            ← insertion index
                _group_name:       ← group name
        entry_0000000001/
        ...
        attrs:
            count: N
    ```

    All nesting from the original `Serializable.dump(recurse=True)` dict is
    encoded into flat dataset/attr names using `→`-separated percent-encoded
    path segments. No sub-groups are created inside entry groups, so every
    `append` performs exactly `(number of arrays)` `create_dataset` calls
    plus two `group.attrs.update` calls, one for scalars and one for metadata.

    **Persistent-handle notes**

    Same as the nested layout. `open(mode="a")` stores the root group in
    `self._handle`; `close()` flushes `_pending_count` and releases it.
    `consolidate` defaults to `False`; pass `close(consolidate=True)`
    when you want fast subsequent reads.
    """

    supports_append: bool = True

    def __init__(
        self,
        store: typing.Union[StoreLike, PathLike, str],
        compressor: typing.Literal["zstd", "lz4", "blosclz"] = "lz4",
        compression_level: int = 1,
        chunks: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> None:
        """
        Initialise the store.

        :param store: Zarr store (file path, directory, or `Store` object).
        :param compressor: Compression algorithm — `'lz4'`, `'zstd'`, or
            `'blosclz'`.
        :param compression_level: Compression level (1-9).
        :param chunks: Optional explicit chunk shape.  When `None` the store
            picks sensible defaults based on array rank.
        :raises StorageError: If the path is invalid or has an incompatible
            extension.
        """
        super().__init__()
        self.store = (
            _validate_filepath(store, is_directory=True, create_if_not_exists=True)
            if isinstance(store, (str, PathLike))
            else store
        )
        self.chunks = chunks
        self._pending_count: int = 0

        # if IS_PYTHON_310_OR_LOWER:
        from numcodecs import Blosc

        self.compressor = Blosc(
            cname=compressor, clevel=compression_level, shuffle=Blosc.BITSHUFFLE
        )
        # else:
        #     from zarr.codecs import BloscCodec, BloscShuffle  # type: ignore[import]

        #     self.compressor = BloscCodec(
        #         cname=compressor,
        #         clevel=compression_level,
        #         shuffle=BloscShuffle.bitshuffle,
        #     )

    def open(self, mode: str = "a", **kwargs: typing.Any) -> None:
        """
        Open the Zarr root group and attach it to `self._handle`.

        :param mode: Zarr open mode — `"a"` (append/create), `"r"`
            (read-only), `"w"` (truncate).
        :raises StorageError: If the group cannot be opened.
        """
        if self._handle is not None:
            return
        try:
            self._handle = zarr.open_group(
                store=self.store,
                mode=mode,  # type: ignore[arg-type]
                zarr_version=2,
            )
            # Seed in-memory counter
            self._pending_count = int(self._handle.attrs.get("count", 0))  # type: ignore
            logger.debug(
                f"{self.__class__.__name__} opened (mode={mode!r}): {self.store!s}"
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to open {self.__class__.__name__}: {exc}"
            ) from exc

    def close(self, consolidate: bool = False) -> None:
        """
        Flush in-memory count` to disk and release the open Zarr handle.

        :param consolidate: If `True`, call `zarr.consolidate_metadata`
            before releasing. False by default. Pass `True` when you want
            faster subsequent reads and are willing to pay the one-time
            directory scan cost.
        """
        if self._handle is None:
            logger.debug(
                f"`{self.__class__.__name__}.open()` called while handle already open; ignored."
            )
            return
        try:
            self._handle.attrs["count"] = self._pending_count
            if consolidate and isinstance(self.store, Path) and self.store.exists():
                try:
                    zarr.consolidate_metadata(self.store)  # type: ignore[arg-type]
                except Exception as exc:
                    logger.error(
                        f"An error occurred while consolidating metadata: {exc}",
                        exc_info=True,
                    )
        except Exception as exc:
            logger.warning(
                f"Error closing {self.__class__.__name__}: {exc}", exc_info=True
            )
        finally:
            self._handle = None
            self._pending_count = 0

    def _get_chunks(
        self, shape: typing.Tuple[int, ...]
    ) -> typing.Optional[typing.Tuple[int, ...]]:
        if self.chunks:
            return self.chunks
        if len(shape) == 3:
            return (min(20, shape[0]), min(20, shape[1]), min(20, shape[2]))
        if len(shape) == 2:
            return (min(100, shape[0]), min(100, shape[1]))
        return (min(shape[0], 1024),)

    def _create_dataset(
        self, group: zarr.Group, name: str, data: np.ndarray
    ) -> zarr.Array:
        chunks = self._get_chunks(data.shape)
        if ZARR_VERSION_GTE_3:
            return group.create_array(
                name=name,
                data=data,
                shape=data.shape,
                chunks=chunks or "auto",
                compressor=self.compressor,
                overwrite=True,
            )
        return group.create_dataset(
            name=name,
            data=data,
            shape=data.shape,
            chunks=chunks,
            compressor=self.compressor,
            overwrite=True,
        )

    def _open_root(self, mode: str) -> zarr.Group:
        """
        Return the active root group.

        Reuses the persistent handle when open; otherwise opens a transient
        group for this call only.
        """
        if self._handle is not None:
            return self._handle
        return zarr.open_group(store=self.store, mode=mode, zarr_version=2)  # type: ignore[arg-type]

    def _write_entry(
        self,
        root: zarr.Group,
        index: int,
        item: SerializableT,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        """
        Write one `Serializable` into *root* at position *index*.

        The entry group contains only flat datasets and two `group.attrs.update`
        calls; one for all scalar values (keyed by encoded path) and one for
        entry metadata. No sub-groups are created.
        """
        group_name = _get_group_name(index)
        item_group = root.require_group(group_name)

        raw = item.dump(recurse=True)
        arrays, scalars, vtypes = _flatten(raw)

        # Collision guard. Two paths must never produce the same key
        assert len(arrays) + len(scalars) == len(set(list(arrays) + list(scalars))), (
            "Flat key collision detected in entry. Field names contain the "
            "separator character after encoding. This is a bug."
        )

        # Write all arrays as flat datasets
        for flat_key, arr in arrays.items():
            self._create_dataset(item_group, name=flat_key, data=arr)

        # Write all scalars and vtypes in two `attrs.update` calls
        if scalars:
            item_group.attrs.update(scalars)

        item_group.attrs.update(
            {
                "_vtypes": vtypes,
                "_meta": meta(item) if meta is not None else {},
                "_index": index,
                "_group_name": group_name,
            }
        )
        return EntryMeta(idx=index, group_name=group_name, meta={})

    def _read_entry(self, item_group: zarr.Group) -> typing.Dict[str, typing.Any]:
        """
        Reconstruct a nested dict from a flat entry group.

        Strips internal metadata keys before returning.
        """
        # Collect flat arrays
        arrays = {
            key: item_group[key][:]  # type: ignore[index]
            for key in item_group.array_keys()  # type: ignore[attr-defined]
        }

        # Collect flat scalars, stripping internal keys
        scalars: typing.Dict[str, typing.Any] = {
            k: v for k, v in item_group.attrs.items() if k not in _INTERNAL
        }
        vtypes: typing.Dict[str, str] = dict(item_group.attrs.get("_vtypes", {}))
        return _unflatten(arrays, scalars, vtypes)  # type: ignore[arg-type]

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[
            typing.Callable[[SerializableT], SerializableT]
        ] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        """
        Persist *data*, always overwriting any existing content.

        :param data: Iterable of `Serializable` instances.
        :param validator: Optional per-item validator/transformer.
        :param meta: Optional callable returning a metadata dict for each item.
        """
        had_open_handle = self._handle is not None
        if had_open_handle:
            # Flush and close existing handle before truncating
            self.close()

        root = zarr.open_group(store=self.store, mode="w", zarr_version=2)  # type: ignore[arg-type]
        count = 0
        for index, item in enumerate(data):
            if validator is not None:
                item = validator(item)
            self._write_entry(root, index, item, meta)
            count += 1

        root.attrs["count"] = count
        logger.debug(
            f"{self.__class__.__name__}: dump complete, {count} entries → {self.store!s}"
        )
        if had_open_handle:
            # Re-open so subsequent operations still work
            self.open(mode="a")

    @reraise_storage_error
    def append(
        self,
        item: SerializableT,
        validator: typing.Optional[
            typing.Callable[[SerializableT], SerializableT]
        ] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        """
        Append a single item without rewriting existing entries.

        :param item: The `Serializable` instance to persist.
        :param validator: Optional validator/transformer applied before writing.
        :param meta: Optional callable returning a metadata dict.
        :returns: The `EntryMeta` record for the appended item.
        """
        root = self._open_root("a")

        if self._handle is not None:
            # Uses the in-memory `_pending_count` when a persistent handle is open
            index = self._pending_count
        else:
            index = int(root.attrs.get("count", 0))  # type: ignore

        if validator is not None:
            item = validator(item)

        entry = self._write_entry(root, index, item, meta)
        if self._handle is not None:
            self._pending_count += 1
        else:
            root.attrs["count"] = index + 1

        logger.debug(f"{self.__class__.__name__}: appended entry {entry.idx}")
        return entry

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        """
        Return metadata for every stored item in insertion order.

        Does not deserialise any payload data.

        :returns: List of `EntryMeta` instances in insertion order.
        """
        try:
            root = self._open_root("r")
        except Exception as exc:
            logger.error(exc, exc_info=True)
            return []

        metas = []
        for name in sorted(root.group_keys()):  # type: ignore[attr-defined]
            idx = _get_index_from_group_name(name)
            if idx is not None:
                group = root[name]
                metas.append(
                    EntryMeta(
                        idx=idx,
                        group_name=name,
                        meta=dict(group.attrs.get("_meta", {})),
                    )
                )
        return metas

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[
            typing.Callable[[SerializableT], SerializableT]
        ] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load and yield items from the store in insertion order.

        Filtering is applied before any array data is deserialised.

        :param typ: The `Serializable` subclass to deserialise into.
        :param indices: If given, load only entries at these positions.
        :param predicate: If given (and `indices` is `None`), yield only
            entries for which `predicate(entry_meta)` returns `True`.
        :param validator: Optional post-load callable applied before yielding.
        :returns: Generator of deserialised items.
        """
        root = self._open_root("r")

        if indices is not None:
            index_set = set(indices)
            for name in sorted(root.group_keys()):  # type: ignore[attr-defined]
                idx = _get_index_from_group_name(name)
                if idx is not None and idx in index_set:
                    raw = self._read_entry(root[name])  # type: ignore[index]
                    obj = typ.load(raw)
                    yield validator(obj) if validator is not None else obj
        else:
            for name in sorted(root.group_keys()):  # type: ignore[attr-defined]
                idx = _get_index_from_group_name(name)
                if idx is not None:
                    group = root[name]
                    entry_meta = EntryMeta(
                        idx=idx,
                        group_name=name,
                        meta=dict(group.attrs.get("_meta", {})),
                    )
                    if predicate is None or predicate(entry_meta):
                        raw = self._read_entry(group)  # type: ignore[arg-type]
                        obj = typ.load(raw)
                        yield validator(obj) if validator is not None else obj

    def flush(self) -> None:
        """Clear every data item stored (destructive)."""
        store = self.store
        if isinstance(store, Path):
            shutil.rmtree(store)

    def __repr__(self) -> str:
        cname = getattr(self.compressor, "cname", str(self.compressor))
        return f"{self.__class__.__name__}(store={self.store!r}, compressor={cname!r})"


@data_store("hdf5", "h5")
class HDF5Store(DataStore[SerializableT, h5py.File]):
    """
    HDF5-based storage.

    Industry standard, good compression, wide tool support.
    May be faster than Zarr for non-parallel workloads.

    Layout:
    ```mermaid
    <file.h5>
        /entry_0000000000      ← one group per item
            <field>            ← dataset  (numpy arrays)
            <nested>/          ← subgroup (mappings / sequences of mappings)
                                    attrs hold scalars, strings, None sentinels
        /entry_0000000001
        ...
        (file attrs: count)
    ```

    **Persistent-handle notes**

    `open(mode="a")` opens the HDF5 file once via `h5py.File` and stores
    the file object in `self._handle`.  All subsequent `append` /
    `load` / `entries` calls reuse the same open file descriptor, avoiding
    the open/close cost on every call.  `close()` flushes and closes the
    `h5py.File` and sets `self._handle = None`.

    Typical high-throughput usage:

    ```python
    with store(mode="a"):
        for state in simulation():
            store.append(state)
    ```

    Note: Calling `dump()` while a handle is open will temporarily close the
    handle (to safely truncate the file), perform the write, then reopen
    it in `"a"` mode so subsequent `append` calls continue to work.
    """

    supports_append: bool = True

    def __init__(
        self,
        filepath: typing.Union[PathLike, str],
        compression: typing.Literal["gzip", "lzf", "szip"] = "gzip",
        compression_opts: int = 3,
        chunks: typing.Optional[typing.Tuple[int, ...]] = None,
    ):
        """
        Initialize the store

        :param filepath: Path to the HDF5 file
        :param compression: Compression algorithm - 'gzip', 'lzf', or 'szip'
        :param compression_opts: Compression level (1-9 for gzip)
        :param chunks: Custom chunk shape for datasets.
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        super().__init__()
        self.filepath = _validate_filepath(
            filepath, expected_extension=".h5", create_if_not_exists=True
        )
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunks = chunks
        # In-memory count cache. Helps us avoid reading/writing file attrs on every append
        # when a persistent handle is open. Only flushed to disk on `close()`.
        self._pending_count: int = 0

    def open(self, mode: str = "a", **kwargs: typing.Any) -> None:
        """
        Open the HDF5 file and attach the `h5py.File` to `self._handle`.

        :param mode: h5py open mode — `"a"` (append/create), `"r"`
            (read-only), `"w"` (truncate).
        :raises StorageError: If the file cannot be opened.
        """
        if self._handle is not None:
            logger.debug(
                f"`{self.__class__.__name__}.open()` called while handle already open; ignored."
            )
            return

        try:
            self._handle = h5py.File(str(self.filepath), mode=mode, **kwargs)
            # Seed the in-memory counter
            self._pending_count = int(self._handle.attrs.get("count", 0))
            logger.debug(
                f"{self.__class__.__name__} opened (mode={mode!r}): {self.filepath!r}"
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to open {self.__class__.__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        """
        Flush the in-memory count to disk and close the open `h5py.File`.

        Idempotent. Safe to call when no handle is open.
        """
        if self._handle is None:
            return

        try:
            self._handle.attrs["count"] = self._pending_count
            self._handle.flush()
            self._handle.close()
            logger.debug(f"{self.__class__.__name__} closed: {self.filepath}")
        except Exception as exc:
            logger.warning(
                f"Error closing {self.__class__.__name__}: {exc}", exc_info=True
            )
        finally:
            self._handle = None
            self._pending_count = 0

    @contextmanager
    def _get_file(self, mode: str) -> typing.Generator[h5py.File, None, None]:
        """
        Yield an open `h5py.File.`

        If a persistent handle is open, yield it directly (ignoring *mode*).
        Otherwise open a transient file, yield it, and close it on exit.
        """
        if self._handle is not None:
            yield self._handle
            return  # Caller owns the handle, hence we must not close it
        else:
            f = h5py.File(str(self.filepath), mode=mode)
            try:
                yield f
            finally:
                f.close()

    def _get_chunks(
        self, shape: typing.Tuple[int, ...]
    ) -> typing.Optional[typing.Tuple[int, ...]]:
        if self.chunks:
            return self.chunks
        if len(shape) == 3:
            return (min(20, shape[0]), min(20, shape[1]), min(20, shape[2]))
        elif len(shape) == 2:
            return (min(100, shape[0]), min(100, shape[1]))
        return None

    def _create_dataset(self, group: h5py.Group, name: str, data: np.ndarray):
        if group.get(name) is not None:
            del group[name]
        chunks = self._get_chunks(shape=data.shape)
        return group.create_dataset(
            name=name,
            data=data,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=chunks if chunks is not None else True,
        )

    def _write_data(self, group: h5py.Group, data: Mapping[str, typing.Any]) -> None:
        for key, value in data.items():
            value = _normalize_for_storage(value)

            if isinstance(value, Mapping):
                sub_group = group.require_group(name=key)
                self._write_data(group=sub_group, data=value)

            elif isinstance(value, np.ndarray):
                if value.dtype == object:
                    raise TypeError(
                        f"HDF5 cannot store object-dtype arrays: {group.name}/{key}"
                    )
                self._create_dataset(group=group, name=key, data=value)

            elif isinstance(value, (np.integer, np.floating)):
                group.attrs[key] = value.item()
            elif isinstance(value, np.bool_):
                group.attrs[key] = bool(value)

            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if not value:
                    self._create_dataset(
                        group=group, name=key, data=np.empty((0,), dtype=np.int8)
                    )
                    continue
                if isinstance(value[0], Mapping):
                    seq_group = group.require_group(key)
                    for i, item in enumerate(value):
                        item_group = seq_group.require_group(str(i))
                        self._write_data(group=item_group, data=item)
                    continue
                array = _sequence_to_ndarray(value=value, path=f"{group.name}/{key}")
                self._create_dataset(group=group, name=key, data=array)

            else:
                group.attrs[key] = value

    def _write_entry(
        self,
        f: h5py.File,
        index: int,
        item: SerializableT,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        group_name = _get_group_name(index)
        item_group = f.require_group(group_name)
        self._write_data(item_group, item.dump(recurse=True))
        item_group.attrs.update(
            {
                "_meta": orjson.dumps(meta(item) if meta is not None else {}),
                "_index": index,
                "_group_name": group_name,
            }
        )
        return EntryMeta(idx=index, group_name=group_name, meta={})

    def _read_entry(self, group: h5py.Group) -> typing.Dict[str, typing.Any]:
        data: typing.Dict[str, typing.Any] = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = _normalize_loaded_value(item[:])  # type: ignore
            elif isinstance(item, h5py.Group):
                loaded = self._read_entry(group=item)
                data[key] = _normalize_loaded_mapping_sequence(loaded)  # type: ignore

        for attr_name in group.attrs:
            data[attr_name] = _denormalize_from_storage(group.attrs[attr_name])
        return data

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        had_open_handle = self._handle is not None
        if had_open_handle:
            # Flush and close existing handle before truncating
            self.close()

        # `dump` always truncates. so we open a dedicated truncating file and never
        # reuse a persistent append handle
        with h5py.File(str(self.filepath), mode="w") as f:
            count = 0
            for index, item in enumerate(data):
                if validator is not None:
                    item = validator(item)
                self._write_entry(f, index, item, meta)
                logger.debug(f"{self.__class__.__name__}: wrote entry {index}")
                count += 1
            f.attrs["count"] = count
        logger.debug(
            f"{self.__class__.__name__}: dump complete, {count} entries → {self.filepath}"
        )

        if had_open_handle:
            # Re-open so subsequent operations still work
            self.open(mode="a")

    @reraise_storage_error
    def append(
        self,
        item: SerializableT,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        mode = "a" if self.filepath.exists() else "w"
        with self._get_file(mode) as f:
            if self._handle is not None:
                # Use the in-memory counter
                index = self._pending_count
            else:
                index = int(f.attrs.get("count", 0))

            if validator is not None:
                item = validator(item)

            entry = self._write_entry(f, index, item, meta)

            if self._handle is not None:
                self._pending_count += 1  # To be flushed to disk on `close()`
            else:
                f.attrs["count"] = index + 1

        logger.debug(f"{self.__class__.__name__}: appended entry {entry.idx}")
        return entry

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        if not self.filepath.exists():
            return []

        metas = []
        with self._get_file("r") as f:
            for name in sorted(f.keys()):
                idx = _get_index_from_group_name(name)
                if idx is not None:
                    group = f[name]
                    metas.append(
                        EntryMeta(
                            idx=idx,
                            group_name=name,
                            meta=orjson.loads(group.attrs.get("_meta", "{}")),
                        )
                    )
        return metas

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        with self._get_file("r") as f:
            if indices is not None:
                index_set = set(indices)
                for name in sorted(f.keys()):
                    idx = _get_index_from_group_name(name)
                    if idx is not None and idx in index_set:
                        item_group = typing.cast(h5py.Group, f[name])
                        logger.debug(f"{self.__class__.__name__}: loading entry {idx}")
                        raw = self._read_entry(item_group)
                        raw.pop("_index", None)
                        raw.pop("_group_name", None)
                        raw.pop("count", None)
                        obj = typ.load(raw)
                        yield validator(obj) if validator is not None else obj
            else:
                for name in sorted(f.keys()):
                    idx = _get_index_from_group_name(name)
                    if idx is not None:
                        group = f[name]
                        entry_meta = EntryMeta(
                            idx=idx,
                            group_name=name,
                            meta=orjson.loads(group.attrs.get("_meta", "{}")),
                        )
                        if predicate is None or predicate(entry_meta):
                            item_group = typing.cast(
                                h5py.Group, f[entry_meta.group_name]
                            )
                            logger.debug(
                                f"{self.__class__.__name__}: loading entry {entry_meta.idx}"
                            )
                            raw = self._read_entry(item_group)
                            raw.pop("_index", None)
                            raw.pop("_group_name", None)
                            raw.pop("count", None)
                            obj = typ.load(raw)
                            yield validator(obj) if validator is not None else obj

    def flush(self) -> None:
        with h5py.File(str(self.filepath), mode="w"):
            pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"filepath={self.filepath!r}, "
            f"compression={self.compression!r}, "
            f"compression_opts={self.compression_opts})"
        )


@data_store("json")
class JSONStore(DataStore[SerializableT, typing.List[typing.Any]]):
    """
    JSON-based storage.  Human-readable, no compression.  Good for configs.

    **Persistent-handle notes**

    JSON files must be read and written in their entirety (there is no
    append-friendly on-disk format), so `open()` loads the current file
    contents into `self._handle` (a plain Python list) and `close()`
    serialises that list back to disk.

    While the handle is open, `dump` replaces `_handle` in memory and
    `append` pushes a new entry onto `_handle`.  Neither touches the
    file until `close()` is called, so you get one write per session
    instead of one write per append:

    ```python
    with store(mode="a"):
        for item in items:
            store.append(item)   # in-memory only
    # ← file written once here by close()
    ```

    Note: `JSONStore.supports_append` is `False` as a class attribute
    (plain append calls without an open handle still rewrite the whole file),
    but the persistent-handle pattern above achieves efficient bulk appending.
    """

    supports_append: bool = False

    def __init__(
        self,
        filepath: typing.Union[PathLike, str],
    ):
        """
        Initialize the store

        :param filepath: Path to the JSON file
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        super().__init__()
        self.filepath = _validate_filepath(
            filepath, expected_extension=".json", create_if_not_exists=True
        )

    def open(self, mode: str = "a", **kwargs: typing.Any) -> None:
        """
        Load the JSON file into memory as `self._handle` (a list).

        :param mode: `"a"` to load existing contents for appending (default),
            `"w"` to start with an empty list (discard existing data).
        :raises StorageError: If the file cannot be read.
        """
        if self._handle is not None:
            logger.debug(
                f"`{self.__class__.__name__}.open()` called while handle already open; ignored."
            )
            return
        try:
            if mode == "w" or not self.filepath.exists():
                self._handle = []
            else:
                with open(self.filepath, "rb") as f:
                    self._handle = orjson.loads(f.read()) if f.read(1) else []
                    # Re-read properly
                with open(self.filepath, "rb") as f:
                    content = f.read()
                    self._handle = orjson.loads(content) if content.strip() else []
            logger.debug(
                f"{self.__class__.__name__} opened (mode={mode!r}): {self.filepath}"
            )
        except Exception as exc:
            self._handle = None
            raise StorageError(
                f"Failed to open {self.__class__.__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        """
        Serialise ``self._handle`` back to disk and release it.

        Idempotent. Safe to call when no handle is open.
        """
        if self._handle is None:
            return
        try:
            with open(self.filepath, mode="wb") as f:
                f.write(safe_json_dumps(self._handle))
            logger.debug(
                f"{self.__class__.__name__} closed (wrote {len(self._handle)} entries): {self.filepath}"
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to close/write {self.__class__.__name__}: {exc}"
            ) from exc
        finally:
            self._handle = None

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        items = []
        for index, item in enumerate(data):
            if validator is not None:
                item = validator(item)
            items.append(
                {
                    "_index": index,
                    "_group_name": _get_group_name(index),
                    "_meta": meta(item) if meta is not None else {},
                    "data": item.dump(recurse=True),
                }
            )

        if self._handle is not None:
            # Replace in-memory list; file written on close()
            self._handle.clear()
            self._handle.extend(items)
        else:
            with open(self.filepath, mode="wb") as f:
                f.write(safe_json_dumps(items))

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        if self._handle is not None:
            items = self._handle
        else:
            if not self.filepath.exists():
                return []
            with open(self.filepath, "rb") as f:
                items = safe_json_loads(f.read())

        return [
            EntryMeta(
                idx=e["_index"],
                group_name=e["_group_name"],
                meta=e["_meta"],
            )
            for e in items
        ]

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        if self._handle is not None:
            items = list(self._handle)
        else:
            with open(self.filepath, mode="rb") as f:
                items = orjson.loads(f.read())

        if indices is not None:
            index_set = set(indices)
            items = [e for e in items if e["_index"] in index_set]
        elif predicate is not None:
            items = [
                e
                for e in items
                if predicate(
                    EntryMeta(
                        idx=e["_index"],
                        group_name=e["_group_name"],
                        meta=e["_meta"],
                    )
                )
            ]

        for entry in items:
            obj = typ.load(entry["data"])
            yield validator(obj) if validator is not None else obj

    def flush(self) -> None:
        self.dump([])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r})"


def _ndarray_representer(
    dumper: typing.Union[yaml.Dumper, yaml.SafeDumper], data: np.typing.NDArray
):
    if data.ndim > 2 or data.size > 50:
        return dumper.represent_mapping(
            "!ndarray",
            {
                "dtype": str(data.dtype),
                "shape": list(data.shape),
                "data": base64.b64encode(data.tobytes()).decode(),
            },
        )
    return dumper.represent_mapping(
        "!ndarray",
        {
            "dtype": str(data.dtype),
            "shape": list(data.shape),
            "data": data.flatten().tolist(),
        },
    )


def _np_scalar_representer(
    dumper: typing.Union[yaml.Dumper, yaml.SafeDumper], data: np.generic
):
    return dumper.represent_mapping(
        "!np_scalar",
        {"dtype": str(data.dtype), "value": data.item()},
    )


def _ndarray_from_base64(
    encoded: str, dtype: np.typing.DTypeLike, shape: typing.Tuple[int, ...]
) -> np.typing.NDArray:
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(shape)


def _ndarray_constructor(
    loader: typing.Union[yaml.Loader, yaml.FullLoader, yaml.UnsafeLoader],
    node: yaml.Node,
):
    try:
        if not isinstance(node, yaml.MappingNode):
            raise TypeError(f"Expected MappingNode, got {type(node)}")
        mapping = loader.construct_mapping(node, deep=True)
        data = mapping["data"]
        dtype = np.dtype(mapping["dtype"])
        shape = tuple(mapping["shape"])
        if isinstance(data, str):
            return _ndarray_from_base64(data, dtype=dtype, shape=shape)
        arr = np.array(data, dtype=dtype)
        if arr.size != np.prod(shape):
            raise ValueError(f"Array size {arr.size} does not match shape {shape}")
        return arr.reshape(shape)
    except Exception:
        print("Failed !ndarray constructor:")
        print(f"  tag: {node.tag}")
        print(
            f"  line: {node.start_mark.line + 1}, column: {node.start_mark.column + 1}"
        )
        print(f"  node type: {type(node).__name__}")
        print(f"  node content: {node.value if hasattr(node, 'value') else node}")
        raise


def _np_scalar_constructor(
    loader: typing.Union[yaml.Loader, yaml.FullLoader, yaml.UnsafeLoader],
    node: yaml.Node,
):
    node = typing.cast(yaml.MappingNode, node)
    mapping = loader.construct_mapping(node, deep=True)
    dtype = np.dtype(mapping["dtype"])
    return dtype.type(mapping["value"])


yaml.add_constructor("!np_scalar", _np_scalar_constructor)
yaml.add_constructor("!ndarray", _ndarray_constructor)

yaml.SafeDumper.add_representer(np.ndarray, _ndarray_representer)
yaml.add_representer(np.ndarray, _ndarray_representer)

yaml.SafeDumper.add_representer(np.generic, _np_scalar_representer)
for _t in [
    np.float16,
    np.float32,
    np.float64,
    np.float128,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]:
    yaml.SafeDumper.add_representer(_t, _np_scalar_representer)
    yaml.add_representer(np.generic, _np_scalar_representer)


@data_store("yaml", "yml")
class YAMLStore(DataStore[SerializableT, typing.List[typing.Any]]):
    """
    YAML-based storage.

    Human-readable format, good for configs, small datasets, and debugging.

    **Persistent-handle notes**

    Like `JSONStore`, YAML files must be read/written as a whole.
    `open()` deserialises the file into `self._handle` (a list) and
    `close()` serialises it back.  All mutations happen in memory; the file
    is touched only on `close()`:

    ```python
    with store(mode="a"):
        for item in items:
            store.append(item)   # in-memory only
    # ← file written once here by `close()`
    ```
    """

    supports_append: bool = False

    def __init__(self, filepath: typing.Union[PathLike, str]):
        """
        Initialize the store

        :param filepath: Path to the YAML file
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        super().__init__()
        self.filepath = _validate_filepath(
            filepath, expected_extension=".yaml", create_if_not_exists=True
        )

    def open(self, mode: str = "a", **kwargs: typing.Any) -> None:
        """
        Load the YAML file into memory as `self._handle` (a list).

        :param mode: `"a"` to load existing contents for appending (default),
            `"w"` to start with an empty list (discard existing data).
        :raises StorageError: If the file cannot be parsed.
        """
        if self._handle is not None:
            logger.debug(
                f"`{self.__class__.__name__}.open()` called while handle already open; ignored."
            )
            return
        try:
            if mode == "w" or not self.filepath.exists():
                self._handle = []
            else:
                with open(self.filepath, mode="r", encoding="utf-8") as f:
                    self._handle = yaml.load(f, Loader=yaml.FullLoader) or []
            logger.debug(
                f"{self.__class__.__name__} opened (mode={mode!r}): {self.filepath}"
            )
        except Exception as exc:
            self._handle = None
            raise StorageError(
                f"Failed to open {self.__class__.__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        """
        Serialise `self._handle` back to disk and release it.

        Idempotent. Safe to call when no handle is open.
        """
        if self._handle is None:
            return
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._handle, f, sort_keys=False)
            logger.debug(
                f"{self.__class__.__name__} closed (wrote {len(self._handle)} entries): {self.filepath}"
            )
        except Exception as exc:
            raise StorageError(
                f"Failed to close/write {self.__class__.__name__}: {exc}"
            ) from exc
        finally:
            self._handle = None

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        items = []
        for index, item in enumerate(data):
            if validator is not None:
                item = validator(item)
            items.append(
                {
                    "_index": index,
                    "_group_name": _get_group_name(index),
                    "_meta": meta(item) if meta is not None else {},
                    "data": item.dump(recurse=True),
                }
            )

        if self._handle is not None:
            self._handle.clear()
            self._handle.extend(items)
        else:
            with open(self.filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(items, f, sort_keys=False)

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        if self._handle is not None:
            items = self._handle
        else:
            if not self.filepath.exists():
                return []
            with open(self.filepath, mode="r", encoding="utf-8") as f:
                items = yaml.load(f, Loader=yaml.FullLoader) or []

        return [
            EntryMeta(
                idx=e["_index"],
                group_name=e["_group_name"],
                meta=e["_meta"],
            )
            for e in items
        ]

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        if self._handle is not None:
            items = list(self._handle)
        else:
            with open(self.filepath, mode="r", encoding="utf-8") as f:
                items = yaml.load(f, Loader=yaml.FullLoader) or []

        if indices is not None:
            index_set = set(indices)
            items = [e for e in items if e["_index"] in index_set]
        elif predicate is not None:
            items = [
                e
                for e in items
                if predicate(
                    EntryMeta(
                        idx=e["_index"],
                        group_name=e["_group_name"],
                        meta=e["_meta"],
                    )
                )
            ]

        for entry in items:
            obj = typ.load(entry["data"])
            yield validator(obj) if validator is not None else obj

    def flush(self) -> None:
        self.dump([])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r})"


def new_store(
    backend: typing.Union[str, typing.Literal["zarr", "hdf5", "json", "yaml"]] = "zarr",
    *args: typing.Any,
    **kwargs: typing.Any,
) -> DataStore:
    """
    Create a new state storage.

    :param backend: Storage backend to use ('zarr', 'hdf5', 'npz', 'pickle', etc.)
    :param args: Additional positional arguments for the store constructor
    :param kwargs: Additional keyword arguments for the store constructor
    :return: An instance of the selected `DataStore` backend

    Example:
    ```python
    store = new_store('zarr', 'simulation.zarr')
    store.dump(states)
    loaded = list(store.load())
    ```
    """
    if backend not in _STORAGE_BACKENDS:
        raise ValidationError(
            f"Unknown backend: {backend}. Available backends are: {list(_STORAGE_BACKENDS.keys())}"
        )

    store_class = _STORAGE_BACKENDS[backend]
    return store_class(*args, **kwargs)


class StoreSerializable(Serializable):
    """Serializable mixin with built-in store/file support."""

    __abstract_serializable__ = True

    @classmethod
    def from_store(
        cls, store: DataStore[Self, typing.Any], **load_kwargs: typing.Any
    ) -> typing.Optional[Self]:
        """
        Load a `Serializable` instance from a `DataStore`.

        :param store: `DataStore` to load the `Serializable` from.
        :return: Loaded `Serializable` instance.
        """
        return next(iter(store.load(cls, **load_kwargs)), None)

    def to_store(
        self, store: DataStore[Self, typing.Any], **dump_kwargs: typing.Any
    ) -> None:
        """
        Dump the `Serializable` instance to a `DataStore`.

        :param store: `DataStore` to dump the `Serializable` to.
        """
        store.dump([self], **dump_kwargs)

    @classmethod
    def from_file(
        cls, filepath: typing.Union[str, PathLike[str]], **load_kwargs: typing.Any
    ) -> typing.Optional[Self]:
        """
        Load a `Serializable` instance from a file.

        :param filepath: Path to the file to load the `Serializable` from.
        :return: Loaded `Serializable` instance.
        """
        path = Path(filepath)
        ext = path.suffix.lower().lstrip(".")
        store = new_store(ext, path)
        return cls.from_store(store, **load_kwargs)

    def to_file(
        self, filepath: typing.Union[str, PathLike[str]], **dump_kwargs: typing.Any
    ) -> None:
        """
        Dump the `Serializable` instance to a file.

        :param filepath: Path to the file to dump the `Serializable` to.
        """
        path = Path(filepath)
        ext = path.suffix.lower().lstrip(".")
        store = new_store(ext, path)
        self.to_store(store, **dump_kwargs)

    save = to_file  # Alias for convenience
