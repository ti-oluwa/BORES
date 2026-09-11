"""HDF5 storage backend"""

import logging
import typing
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from os import PathLike

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import orjson

from bores.errors import StorageError
from bores.serde.base import SerializableT
from bores.serde.stores.base import (
    DataStore,
    DataValidator,
    EntryMeta,
    _get_group_name,
    _get_index_from_group_name,
    reraise_storage_error,
    validate_path,
)
from bores.serde.stores.utils import (
    denormalize_from_storage,
    normalize_for_storage,
    normalize_loaded_value,
    sequence_to_ndarray,
)

__all__ = ["HDF5Store"]

logger = logging.getLogger(__name__)


def _hdf5_string_array(value: typing.Sequence[str]) -> npt.NDArray:
    """
    HDF5's native variable-length string representation
    (`h5py.string_dtype`), falling back to fixed-width unicode (`'U'`) on
    older `h5py` that lacks it.
    """
    if hasattr(h5py, "string_dtype"):
        return np.asarray(value, dtype=h5py.string_dtype(encoding="utf-8"))
    return np.asarray(value, dtype="U")


def _sequence_to_ndarray(value: Sequence, path: str) -> npt.NDArray:
    return sequence_to_ndarray(value, path, string_array_factory=_hdf5_string_array)


def _normalize_loaded_mapping_sequence(value: typing.Any) -> typing.Any:
    """
    Detect if a loaded mapping represents a sequence and convert it back.

    WellGroups with numeric string keys "0", "1", "2", ... are sequences.
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


class HDF5Store(DataStore[SerializableT, h5py.File]):
    """
    HDF5-based storage.

    Industry standard, good compression, wide tool support.
    May be faster than Zarr for non-parallel workloads.

    Layout:
    ```mermaid
    <file.h5>
        /entry_0000000000      <- one group per item
            <field>            <- dataset  (numpy arrays)
            <nested>/          <- subgroup (mappings / sequences of mappings)
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

    can_append: bool = True

    def __init__(
        self,
        filepath: PathLike | str,
        compression: typing.Literal["gzip", "lzf", "szip"] = "gzip",
        compression_opts: int = 3,
        chunks: tuple[int, ...] | None = None,
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
        self.filepath = validate_path(
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

        :param mode: h5py open mode - `"a"` (append/create), `"r"`
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
            logger.debug(f"{self.__class__.__name__} opened (mode={mode!r}): {self.filepath!r}")
        except Exception as exc:
            raise StorageError(f"Failed to open {self.__class__.__name__}: {exc}") from exc

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
            logger.warning(f"Error closing {self.__class__.__name__}: {exc}", exc_info=True)
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

    def _get_chunks(self, shape: tuple[int, ...]) -> tuple[int, ...] | None:
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
            value = normalize_for_storage(value)

            if isinstance(value, Mapping):
                sub_group = group.require_group(name=key)
                self._write_data(group=sub_group, data=value)

            elif isinstance(value, np.ndarray):
                if value.dtype == object:
                    raise TypeError(f"HDF5 cannot store object-dtype arrays: {group.name}/{key}")
                self._create_dataset(group=group, name=key, data=value)

            elif isinstance(value, (np.integer, np.floating)):
                group.attrs[key] = value.item()
            elif isinstance(value, np.bool_):
                group.attrs[key] = bool(value)

            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if not value:
                    self._create_dataset(group=group, name=key, data=np.empty((0,), dtype=np.int8))
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
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
    ) -> EntryMeta:
        group_name = _get_group_name(index)
        item_group = f.require_group(group_name)
        self._write_data(item_group, item.dump())
        item_group.attrs.update({
            "_meta": orjson.dumps(meta(item) if meta is not None else {}),
            "_index": index,
            "_group_name": group_name,
        })
        return EntryMeta(idx=index, group_name=group_name, meta={})

    def _read_entry(self, group: h5py.Group) -> dict[str, typing.Any]:
        data: dict[str, typing.Any] = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = normalize_loaded_value(item[:])  # type: ignore
            elif isinstance(item, h5py.Group):
                loaded = self._read_entry(group=item)
                data[key] = _normalize_loaded_mapping_sequence(loaded)  # type: ignore

        for attr_name in group.attrs:
            data[attr_name] = denormalize_from_storage(group.attrs[attr_name])
        return data

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: DataValidator[SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
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
            f"{self.__class__.__name__}: dump complete, {count} entries -> {self.filepath}"
        )

        if had_open_handle:
            # Re-open so subsequent operations still work
            self.open(mode="a")

    @reraise_storage_error
    def append(
        self,
        item: SerializableT,
        validator: DataValidator[SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
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
    def entries(self) -> list[EntryMeta]:
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
        typ: type[SerializableT],
        indices: typing.Sequence[int] | None = None,
        predicate: typing.Callable[[EntryMeta], bool] | None = None,
        validator: DataValidator[SerializableT] | None = None,
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
                            item_group = typing.cast(h5py.Group, f[entry_meta.group_name])
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
