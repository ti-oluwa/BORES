"""YAML storage backend"""

import base64
import logging
import typing
from os import PathLike

import numpy as np
import numpy.typing as npt
import yaml

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

__all__ = ["YAMLStore"]


logger = logging.getLogger(__name__)


def _ndarray_representer(dumper: yaml.Dumper | yaml.SafeDumper, data: npt.NDArray):
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


def _np_scalar_representer(dumper: yaml.Dumper | yaml.SafeDumper, data: np.generic):
    return dumper.represent_mapping(
        "!np_scalar",
        {"dtype": str(data.dtype), "value": data.item()},
    )


def _ndarray_from_base64(
    encoded: str, dtype: npt.DTypeLike, shape: tuple[int, ...]
) -> npt.NDArray:
    raw = base64.b64decode(encoded)
    array = np.frombuffer(raw, dtype=dtype)
    return array.reshape(shape)


def _ndarray_constructor(
    loader: yaml.Loader | yaml.FullLoader | yaml.UnsafeLoader,
    node: yaml.Node,
):
    try:
        if not isinstance(node, yaml.MappingNode):
            raise TypeError(f"Expected `yaml.MappingNode`, got `{type(node)}`")

        mapping = loader.construct_mapping(node, deep=True)
        data = mapping["data"]
        dtype = np.dtype(mapping["dtype"])
        shape = tuple(mapping["shape"])
        if isinstance(data, str):
            return _ndarray_from_base64(data, dtype=dtype, shape=shape)
        array = np.array(data, dtype=dtype)
        if array.size != np.prod(shape):
            raise ValueError(f"Array size {array.size} does not match shape {shape}")
        return array.reshape(shape)
    except Exception:
        print("Failed !ndarray constructor:")
        print(f"  tag: {node.tag}")
        print(f"  line: {node.start_mark.line + 1}, column: {node.start_mark.column + 1}")
        print(f"  node type: {type(node).__name__}")
        print(f"  node content: {node.value if hasattr(node, 'value') else node}")
        raise


def _np_scalar_constructor(
    loader: yaml.Loader | yaml.FullLoader | yaml.UnsafeLoader,
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


class YAMLStore(DataStore[SerializableT, list[typing.Any]]):
    """
    YAML-based storage.

    Human-readable format, good for configs, small datasets, and debugging.

    **Persistent-handle notes**

    Like `JSONStore`, YAML files must be read/written as a whole.
    `open()` deserialises the file into `self._handle` (a list) and
    `close()` serialises it back. All mutations happen in memory; the file
    is touched only on `close()`:

    ```python
    with store(mode="a"):
        for item in items:
            store.append(item)  # in-memory only
    # <- file written once here by `close()`
    ```
    """

    can_append: bool = False

    def __init__(self, filepath: PathLike | str):
        """
        Initialize the store

        :param filepath: Path to the YAML file
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        super().__init__()
        self.filepath = validate_path(
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
                with open(self.filepath, encoding="utf-8") as f:
                    self._handle = yaml.load(f, Loader=yaml.FullLoader) or []
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
            with open(self.filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._handle, f, sort_keys=False)
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
            self._handle.clear()
            self._handle.extend(items)
        else:
            with open(self.filepath, "w", encoding="utf-8") as f:
                yaml.safe_dump(items, f, sort_keys=False)

    @reraise_storage_error
    def entries(self) -> list[EntryMeta]:
        if self._handle is not None:
            items = self._handle
        else:
            if not self.filepath.exists():
                return []
            with open(self.filepath, encoding="utf-8") as f:
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
        typ: type[SerializableT],
        indices: typing.Sequence[int] | None = None,
        predicate: typing.Callable[[EntryMeta], bool] | None = None,
        validator: DataValidator[SerializableT] | None = None,
    ) -> typing.Generator[SerializableT, None, None]:
        if self._handle is not None:
            items = list(self._handle)
        else:
            with open(self.filepath, encoding="utf-8") as f:
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
