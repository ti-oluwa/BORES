import functools
import logging
import typing
from abc import ABC, abstractmethod
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

from typing_extensions import ParamSpec, Self

from bores.errors import StorageError
from bores.serde.base import Serializable, SerializableT

__all__ = ["DataStore", "StoreSerializable"]


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
    """0-based position in insertion order."""

    group_name: str
    """Internal storage key (opaque to callers)."""

    meta: dict[str, str]
    """JSON serializable metadata dictionary"""


class DataStore(ABC, typing.Generic[SerializableT, HandleT]):
    """
    Abstract base class for all storage backends.

    Every backend maintains a compact metadata index (`list[EntryMeta]`) so
    callers can inspect stored entries and jump directly to specific ones without
    a full scan. Group naming is internal and fixed - callers never supply it.
    All writes overwrite existing content.

    **Persistent handle (open / close / __call__)**

    By default every method (`dump`, `load`, `append`, `entries`) opens
    the underlying file/directory, performs its work, and closes it again. For
    workloads that call `append` in a tight loop (e.g. a background I/O
    thread) this per-call overhead is significant.

    Call `open(**kwargs)` once to obtain a persistent handle that all
    subsequent methods will reuse. Call `close()` when finished. The
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
        Persist an iterable of `Serializable` items. Always overwrites.

    `load(typ)`
        Load every item. Returns a generator.

    `load(typ, indices=[0, 3, 7])`
        Load only the items at the given positional indices.

    `load(typ, predicate=lambda e: e.idx < 10)`
        Load only items whose `EntryMeta` satisfies *predicate*.

    `entries()`
        Return the full `list[EntryMeta]` without deserialising any payload.
        Uses this for `count()`, `max_index()`, membership checks, etc.
    """

    can_append: bool = False

    def __init__(self) -> None:
        self._handle: HandleT | None = None
        """The open handle. `None` means "no persistent handle; open/close per call"."""

    @abstractmethod
    def open(self, **kwargs: typing.Any) -> None:
        """
        Opens the underlying storage and attaches the handle to `self._handle`.

        Subsequent calls to `dump`, `load`, `append`, and `entries`
        will use this handle instead of opening the file themselves.

        :param kwargs: Backend-specific keyword arguments (e.g. `mode="a"`).
        :raises StorageError: If the store cannot be opened.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Flushes and release the persistent handle (`self._handle`).

        After this call `self._handle` must be `None`. It is safe to call
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
        validator: DataValidator[SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
    ) -> None:
        """
        Persists *data*, always overwriting any existing content in the store.

        Every item in *data* is written in iteration order. If the backing
        file or directory already exists it is truncated first, so calling
        `dump` twice is equivalent to calling it once with the second dataset.
        Use `append` when you need to add items to an existing store without
        discarding what is already there.

        :param data: Iterable of `Serializable` instances to persist.
        :param validator: Optional callable applied to each item before it is
            written. Receives the item and must return a (possibly transformed)
            item of the same type. Raise to abort persistence of that item.
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
        typ: type[SerializableT],
        indices: typing.Sequence[int] | None = None,
        predicate: typing.Callable[[EntryMeta], bool] | None = None,
        validator: DataValidator[SerializableT] | None = None,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Loads and yield items from the store in insertion order.

        Filtering is applied before any array data is deserialised, so entries
        that do not match have no I/O cost beyond reading their metadata.
        When both `indices` and `predicate` are supplied, `indices` takes
        priority and `predicate` is ignored.

        :param typ: The `Serializable` subclass to deserialise each entry into.
        :param indices: If given, load only the entries at these zero-based
            insertion-order positions. Out-of-range indices raise `IndexError`.
        :param predicate: If given (and `indices` is `None`), yield only entries
            for which `predicate(entry_meta)` returns `True`. The predicate
            receives an `EntryMeta` instance and may inspect `entry_meta.meta`
            to filter on stored metadata without touching array data.
        :param validator: Optional callable applied to each deserialised item
            before it is yielded. Receives the item and must return a
            (possibly transformed) item of the same type.
        :return: Generator yielding deserialised items matching the filter.
        """
        ...

    def append(
        self,
        item: SerializableT,
        validator: DataValidator[SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
    ) -> EntryMeta:
        """
        Appends a single item to the store without rewriting existing entries.

        The item is assigned the next available insertion-order index. Unlike
        `dump`, existing entries are never touched. Backends that do not
        support append-style writes raise `NotImplementedError` at call time
        rather than at construction time.

        Check `can_append` before calling if in doubt.

        :param item: The `Serializable` instance to persist.
        :param validator: Optional callable applied to *item* before it is
            written. Receives the item and must return a (possibly transformed)
            item of the same type. Raise to abort the write.
        :param meta: Optional callable that receives *item* and returns a plain
            `dict` of JSON-serialisable values stored on `EntryMeta.meta`.
            Use this to record lightweight metadata (e.g. `{"step": state.step}`)
            that can later be used to filter entries via `load(predicate=...)`
            or `entries()` without deserialising array data.
        :return: The `EntryMeta` record created for the appended item, including
            its assigned index, group name, and any stored metadata.
        :raises NotImplementedError: If the backend does not support appending.
        """
        raise NotImplementedError(f"{self.__class__.__name__!r} does not implement `append(...)`")

    @abstractmethod
    def entries(self) -> list[EntryMeta]:
        """
        Returns metadata for every stored item in insertion order.

        This method must not deserialise any payload data. Implementations
        should read only group names, file keys, and lightweight attributes,
        never array datasets. The returned list can therefore be used for
        cheap introspection (counts, step lookups, predicate filtering) without
        triggering any significant I/O.

        :return: List of `EntryMeta` instances in insertion order.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flushes the data store clean. Clear every data item stored."""
        ...

    def count(self) -> int:
        """
        Returns the number of items currently stored.

        Delegates to `entries()` and returns its length. No payload data is
        deserialised.

        :return: Total number of stored entries.
        """
        return len(self.entries())

    def max_index(self) -> int | None:
        """
        Returns the highest insertion-order index in the store, or `None` if empty.

        Useful for targeting the last written entry via
        `store.load(typ, indices=[store.max_index()])` without replaying
        the entire store. No payload data is deserialised.

        :return: The highest `EntryMeta.index` value, or `None` if the store
            contains no entries.
        """
        metas = self.entries()
        return max(e.idx for e in metas) if metas else None


def validate_path(
    filepath: PathLike | str,
    expected_extension: str | None = None,
    is_directory: bool = False,
    create_if_not_exists: bool = False,
) -> Path:
    """
    Validates and normalizes a filepath for state storage.

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


def _get_group_name(index: int) -> str:
    """
    Returns a group name using fixed naming scheme: `entry_{index:010d}`.

    Zero-padded to 10 digits so lexicographic order == insertion order,
    meaning `sorted(keys)` always gives the correct traversal order.
    """
    return f"entry_{index:010d}"


def _get_index_from_group_name(name: str) -> int | None:
    """Parses group name of form `entry_NNNNNNNNNN` to integer index, or `None` if not our format."""
    if name.startswith("entry_") and len(name) == 16:
        try:
            return int(name[6:])
        except ValueError:
            return None
    return None


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


class StoreSerializable(Serializable):
    """Serializable mixin with built-in store/file support."""

    __abstract_serializable__ = True

    @classmethod
    def read(cls, store: DataStore[Self, typing.Any], **load_kwargs: typing.Any) -> Self | None:
        """
        Reads and loads a `Serializable` instance from a `DataStore`.

        :param store: `DataStore` to load the `Serializable` from.
        :return: Loaded `Serializable` instance.
        """
        return next(iter(store.load(cls, **load_kwargs)), None)

    def save(self, store: DataStore[Self, typing.Any], **dump_kwargs: typing.Any) -> None:
        """
        Dumps and saves the `Serializable` instance to a `DataStore`.

        :param store: `DataStore` to dump the `Serializable` to.
        """
        store.dump([self], **dump_kwargs)
