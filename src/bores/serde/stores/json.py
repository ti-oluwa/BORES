"""JSON storage backend."""

import logging
import typing
from os import PathLike

import orjson

from bores.errors import StorageError
from bores.serde.base import SerializableT
from bores.serde.stores.base import (
    DataStore,
    DataValidator,
    EntryMeta,
    _get_group_name,
    reraise_storage_error,
    validate_path,
)
from bores.utils import safe_json_dumps, safe_json_loads

__all__ = ["JSONStore"]

logger = logging.getLogger(__name__)


class JSONStore(DataStore[SerializableT, list[typing.Any]]):
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
            store.append(item)  # in-memory only
    # <- file written once here by close()
    ```

    Note: `JSONStore.can_append` is `False` as a class attribute
    (plain append calls without an open handle still rewrite the whole file),
    but the persistent-handle pattern above achieves efficient bulk appending.
    """

    can_append: bool = False

    def __init__(
        self,
        filepath: PathLike | str,
    ):
        """
        Initialize the store

        :param filepath: Path to the JSON file
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        super().__init__()
        self.filepath = validate_path(
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
                    content = f.read()
                self._handle = orjson.loads(content) if content.strip() else []
            logger.debug(f"{self.__class__.__name__} opened (mode={mode!r}): {self.filepath}")
        except Exception as exc:
            self._handle = None
            raise StorageError(f"Failed to open {self.__class__.__name__}: {exc}") from exc

    def close(self) -> None:
        """
        Serialise `self._handle` back to disk and release it.

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
            raise StorageError(f"Failed to close/write {self.__class__.__name__}: {exc}") from exc
        finally:
            self._handle = None

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: DataValidator[SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
    ) -> None:
        items = []
        for index, item in enumerate(data):
            if validator is not None:
                item = validator(item)
            items.append({
                "_index": index,
                "_group_name": _get_group_name(index),
                "_meta": meta(item) if meta is not None else {},
                "data": item.dump(),
            })

        if self._handle is not None:
            # Replace in-memory list; file written on close()
            self._handle.clear()
            self._handle.extend(items)
        else:
            with open(self.filepath, mode="wb") as f:
                f.write(safe_json_dumps(items))

    @reraise_storage_error
    def entries(self) -> list[EntryMeta]:
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
        typ: type[SerializableT],
        indices: typing.Sequence[int] | None = None,
        predicate: typing.Callable[[EntryMeta], bool] | None = None,
        validator: DataValidator[SerializableT] | None = None,
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
