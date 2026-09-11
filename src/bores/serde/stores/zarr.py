"""Zarr storage backend"""

import logging
import shutil
import sys
import typing
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson
import zarr  # type: ignore[import-untyped]
from numcodecs.vlen import VLenUTF8
from packaging.version import Version
from zarr.storage import StoreLike  # type: ignore[import-untyped]

from bores.errors import StorageError
from bores.serde.base import SerializableT
from bores.serde.stores.base import (
    DataStore,
    EntryMeta,
    _get_group_name,
    _get_index_from_group_name,
    reraise_storage_error,
    validate_path,
)
from bores.serde.stores.utils import (
    denormalize_from_storage,
    normalize_loaded_value,
    sequence_to_ndarray,
)

__all__ = ["ZarrStore"]

IS_PYTHON_310_OR_LOWER = sys.version_info < (3, 11)
ZARR_VERSION_GTE_3 = Version(version("zarr")) >= Version("3.0.0")


logger = logging.getLogger(__name__)


# Sentinel for None values to avoid object dtype issues
_NONE_SENTINEL = "__NONE_SENTINEL__"


def _zarr_string_array(value: typing.Sequence[str]) -> npt.NDArray:
    """
    Zarr's native variable-length string representation: plain
    `dtype=object`. No `h5py` dependency - `_create_dataset` recognizes
    `dtype=object` and routes it to `zarr`'s own string handling per
    version: `dtype=str` (zarr's own `StringDType` resolution) on Zarr 3,
    `object_codec=numcodecs.VLenUTF8()` (already a direct dependency via
    `Blosc`) on Zarr 2. Confirmed working on both, verified by testing -
    `h5py` was never actually required for this, it was carried over from
    sharing code with `hdf5.py`.
    """
    return np.asarray(value, dtype=object)


def _sequence_to_ndarray(value: Sequence, path: str) -> npt.NDArray:
    return sequence_to_ndarray(value, path, string_array_factory=_zarr_string_array)


"""
`DataStore` with flattened entry layout.

**Encoding contract**
Path segments are percent-encoded so that the separator character (``->``,
U+2192) and the escape character (``%``) can never appear unescaped in a
segment.  This makes path encoding injective: two distinct nested paths can
never produce the same flat key.

    encode: "%" -> "%25",  "->" -> "%E2%86%92"
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


_SEP = "\u2192"  # -> U+2192  RIGHTWARDS ARROW - path segment separator
_ESC = "%"  # percent  - escape character

_SEP_ENCODED = "%E2%86%92"
_ESC_ENCODED = "%25"


def _encode_segment(s: str) -> str:
    """Percent-encode `%` and `->` so neither can appear raw in a segment."""
    return s.replace(_ESC, _ESC_ENCODED).replace(_SEP, _SEP_ENCODED)


def _decode_segment(s: str) -> str:
    """Reverse of `_encode_segment`."""
    return s.replace(_SEP_ENCODED, _SEP).replace(_ESC_ENCODED, _ESC)


def _join_path(*segments: str) -> str:
    return _SEP.join(_encode_segment(s) for s in segments)


def _split_path(flat_key: str) -> list[str]:
    return [_decode_segment(s) for s in flat_key.split(_SEP)]


_VTYPE_JSON = "json"  # list-of-mappings or arbitrary list/dict -> orjson
_VTYPE_NONE = "none"  # Python None
_VTYPE_BOOL = "bool"  # Python bool

_INTERNAL = {"_vtypes", "_meta", "_index", "_group_name", "count", "version"}


def _flatten(
    data: typing.Mapping[str, typing.Any],
    prefix: tuple[str, ...] = (),
    out_arrays: dict[str, npt.NDArray] | None = None,
    out_scalars: dict[str, typing.Any] | None = None,
    out_vtypes: dict[str, str] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, typing.Any],
    dict[str, str],
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
        path = (*prefix, key)
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

        # Convert numpy scalars to Python native
        if isinstance(value, (np.integer, np.floating)):
            out_scalars[flat_key] = value.item()
            continue

        if isinstance(value, np.bool_):
            out_scalars[flat_key] = bool(value)
            out_vtypes[flat_key] = _VTYPE_BOOL
            continue

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                # Empty sequence -> zero-length dataset
                out_arrays[flat_key] = np.empty((0,), dtype=np.int8)
                continue

            if isinstance(value[0], Mapping):
                # List of mappings -> serialise as JSON scalar; not worth
                # the complexity of flattening further since these are always
                # small (well perforations, schedule entries, etc.)
                out_scalars[flat_key] = orjson.dumps(
                    [dict(item) for item in value], option=orjson.OPT_SERIALIZE_NUMPY
                ).decode()
                out_vtypes[flat_key] = _VTYPE_JSON
                continue

            # Homogeneous sequence of scalars/arrays -> convert to ndarray
            array = _sequence_to_ndarray(value, path=flat_key)
            out_arrays[flat_key] = array
            continue

        if isinstance(value, bool):
            out_scalars[flat_key] = value
            out_vtypes[flat_key] = _VTYPE_BOOL
            continue

        # other scalars, int, float, str, etc.
        out_scalars[flat_key] = value

    return out_arrays, out_scalars, out_vtypes


def _unflatten(
    arrays: dict[str, npt.NDArray],
    scalars: dict[str, typing.Any],
    vtypes: dict[str, str],
) -> dict[str, typing.Any]:
    """
    Reconstruct a nested dict from the three flat dicts produced by `_flatten`.

    :param arrays: `{flat_key: ndarray}` - from zarr array datasets.
    :param scalars: `{flat_key: scalar}` - from zarr group attrs.
    :param vtypes: `{flat_key: vtype_tag}` - from the `_vtypes` attr.
    :returns: Reconstructed nested dict.
    """
    result: dict[str, typing.Any] = {}

    def _set_nested(d: dict, parts: list[str], value: typing.Any) -> None:
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

    for flat_key, array in arrays.items():
        parts = _split_path(flat_key)
        value = normalize_loaded_value(array)
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
            value = denormalize_from_storage(raw)

        _set_nested(result, parts, value)
    return result


class ZarrStore(DataStore[SerializableT, zarr.Group]):
    """
    Zarr-based storage.

    Fast, efficient compression with lazy loading.
    Good for large 3D numpy arrays.

    **Layout**

    ```mermaid
    <root.zarr>/
        entry_0000000000/          <- one group per item
            <encoded->path>         <- zarr dataset  (numpy arrays)
            attrs:
                <encoded->path>     <- scalar values
                _scalars_encoded->path: value
                _vtypes:           <- type tags for non-trivial scalars
                _meta:             <- user metadata
                _index:            <- insertion index
                _group_name:       <- group name
        entry_0000000001/
        ...
        attrs:
            count: N
    ```

    All nesting from the original `Serializable.dump()` dict is
    encoded into flat dataset/attr names using `->`-separated percent-encoded
    path segments. No sub-groups are created inside entry groups, so every
    `append` performs exactly `(number of arrays)` `create_dataset` calls
    plus two `group.attrs.update` calls, one for scalars and one for metadata.

    **Persistent-handle notes**

    Same as the nested layout. `open(mode="a")` stores the root group in
    `self._handle`; `close()` flushes `_pending_count` and releases it.
    `consolidate` defaults to `False`; pass `close(consolidate=True)`
    when you want fast subsequent reads.
    """

    can_append: bool = True

    def __init__(
        self,
        store: StoreLike | PathLike | str,
        compressor: typing.Literal["zstd", "lz4", "blosclz"] = "lz4",
        compression_level: int = 1,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialise the store.

        :param store: Zarr store (file path, directory, or `Store` object).
        :param compressor: Compression algorithm - `'lz4'`, `'zstd'`, or
            `'blosclz'`.
        :param compression_level: Compression level (1-9).
        :param chunks: Optional explicit chunk shape.  When `None` the store
            picks sensible defaults based on array rank.
        :raises StorageError: If the path is invalid or has an incompatible
            extension.
        """
        super().__init__()
        self.store = (
            validate_path(store, is_directory=True, create_if_not_exists=True)
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

        :param mode: Zarr open mode - `"a"` (append/create), `"r"`
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
            logger.debug(f"{self.__class__.__name__} opened (mode={mode!r}): {self.store!s}")
        except Exception as exc:
            raise StorageError(f"Failed to open {self.__class__.__name__}: {exc}") from exc

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
            logger.warning(f"Error closing {self.__class__.__name__}: {exc}", exc_info=True)
        finally:
            self._handle = None
            self._pending_count = 0

    def _get_chunks(self, shape: tuple[int, ...]) -> tuple[int, ...] | None:
        if self.chunks:
            return self.chunks
        if len(shape) == 3:
            return (min(20, shape[0]), min(20, shape[1]), min(20, shape[2]))
        if len(shape) == 2:
            return (min(100, shape[0]), min(100, shape[1]))
        return (min(shape[0], 1024),)

    def _create_dataset(self, group: zarr.Group, name: str, data: np.ndarray) -> zarr.Array:
        chunks = self._get_chunks(data.shape)
        is_string_array = data.dtype == object
        # `_sequence_to_ndarray`/`_zarr_string_array` only ever produce
        # `dtype=object` for the "sequence of str" case (every other path
        # raises rather than falling back to `object`), so this is an
        # unambiguous signal, not a heuristic.
        if ZARR_VERSION_GTE_3:
            return group.create_array(  # type: ignore
                name=name,
                data=data,
                dtype=str if is_string_array else data.dtype,
                chunks=chunks or "auto",
                compressor=self.compressor,
                overwrite=True,
            )
        return group.create_dataset(
            name=name,
            data=data,
            shape=data.shape,
            dtype=object if is_string_array else data.dtype,
            object_codec=VLenUTF8() if is_string_array else None,
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
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
    ) -> EntryMeta:
        """
        Write one `Serializable` into *root* at position *index*.

        The entry group contains only flat datasets and two `group.attrs.update`
        calls; one for all scalar values (keyed by encoded path) and one for
        entry metadata. No sub-groups are created.
        """
        group_name = _get_group_name(index)
        item_group = root.require_group(group_name)

        raw = item.dump()
        arrays, scalars, vtypes = _flatten(raw)

        # Collision guard. Two paths must never produce the same key
        assert len(arrays) + len(scalars) == len(set(list(arrays) + list(scalars))), (
            "Flat key collision detected in entry. Field names contain the "
            "separator character after encoding. This is a bug."
        )

        # Write all arrays as flat datasets
        for flat_key, array in arrays.items():
            self._create_dataset(item_group, name=flat_key, data=array)

        # Write all scalars and vtypes in two `attrs.update` calls
        if scalars:
            item_group.attrs.update(scalars)

        item_group.attrs.update({
            "_vtypes": vtypes,
            "_meta": meta(item) if meta is not None else {},
            "_index": index,
            "_group_name": group_name,
        })
        return EntryMeta(idx=index, group_name=group_name, meta={})

    def _read_entry(self, item_group: zarr.Group) -> dict[str, typing.Any]:
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
        scalars: dict[str, typing.Any] = {
            k: v for k, v in item_group.attrs.items() if k not in _INTERNAL
        }
        vtypes: dict[str, str] = dict(item_group.attrs.get("_vtypes", {}))
        return _unflatten(arrays, scalars, vtypes)  # type: ignore[arg-type]

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Callable[[SerializableT], SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
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
            f"{self.__class__.__name__}: dump complete, {count} entries -> {self.store!s}"
        )
        if had_open_handle:
            # Re-open so subsequent operations still work
            self.open(mode="a")

    @reraise_storage_error
    def append(
        self,
        item: SerializableT,
        validator: typing.Callable[[SerializableT], SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
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
    def entries(self) -> list[EntryMeta]:
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
        typ: type[SerializableT],
        indices: typing.Sequence[int] | None = None,
        predicate: typing.Callable[[EntryMeta], bool] | None = None,
        validator: typing.Callable[[SerializableT], SerializableT] | None = None,
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
