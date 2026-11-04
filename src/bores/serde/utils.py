import base64
import typing

import numpy as np
import numpy.typing as npt

SPARSE_DENSITY_THRESHOLD = 0.05  # < 5% non-fill cells means array is sparse
MIN_SPARSE_CELLS = 10  # Do not bother with sparse on tiny arrays


def _b64_encode(array: npt.NDArray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _b64_decode(s: str, dtype: npt.DTypeLike, shape: tuple[int, ...]) -> npt.NDArray:
    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _sniff_scalar(array: npt.NDArray) -> bool:
    """Returns all values bit-identical to array.flat[0]."""
    return bool(np.all(array == array.flat[0]))


def _sniff_layered(array: npt.NDArray):
    for axis in (2, 1, 0):
        if axis >= array.ndim:
            continue
        # Move target axis to front, then check if each slice is uniform
        moved = np.moveaxis(array, axis, 0)
        layer_values = []
        uniform = True
        for idx in range(moved.shape[0]):
            sl = moved[idx]  # view, no copy
            v0 = sl.flat[0]
            if not np.all(sl == v0):
                uniform = False
                break
            layer_values.append(v0)
        if uniform:
            return axis, np.array(layer_values, dtype=array.dtype)
    return None


def _sniff_sparse(
    array: npt.NDArray,
    *,
    min_sparse_cells: int = 10,
    sparse_density_threshold: float = 0.05,
) -> tuple[typing.Any, npt.NDArray, npt.NDArray] | None:
    """
    Returns (fill_value, flat_indices_int32, values) if the array is sparse,
    else None. Fill value is the most common scalar value (exact, lossless).

    Uses O(N) candidate-probe approach instead of O(N log N) sort:
    sample ~20 evenly-spaced positions as fill-value candidates, then verify
    each with a single O(N) count pass. If the fill covers >95% of elements,
    at least one probe will hit it (probability > 1 - 0.05^20 ≈ 1 - 10^-26).
    """
    if array.size < min_sparse_cells:
        return None

    flat = array.ravel()
    n = flat.size
    max_non_fill = int(n * sparse_density_threshold)

    # Sample evenly-spaced positions as fill-value candidates.
    n_probes = min(20, n)
    step = max(1, n // n_probes)
    candidates = np.unique(flat[::step])

    for candidate in candidates:
        non_fill_count = np.count_nonzero(flat != candidate)
        if non_fill_count <= max_non_fill:
            non_fill_mask = flat != candidate
            indices = np.where(non_fill_mask)[0].astype(np.int32)
            values = flat[non_fill_mask]
            return candidate, indices, values

    return None


def serialize_ndarray(
    array: npt.ArrayLike,
    *,
    min_sparse_cells: int = 10,
    sparse_density_threshold: float = 0.05,
) -> dict[str, typing.Any]:
    """
    Smart serializer for numpy arrays.

    Tries encodings in order: `scalar` -> `layered` -> `sparse` -> `dense`.
    Falls back to `dense` (base64 raw bytes) when no compression applies.
    Ensure that all encodings are exact and there are no approximation.

    Wire format is a dict with `'__ndarray__': True` and an `encoding` key
    so `deserialize_ndarray` can dispatch correctly.
    """
    a = np.asarray(array)
    dtype = a.dtype
    shape = list(a.shape)
    base = {"__ndarray__": True, "dtype": dtype.str, "shape": shape}

    # Handle empty arrays before any access to elements
    if a.size == 0:
        return {**base, "encoding": "empty"}

    if _sniff_scalar(a):
        # Store the fill value as a Python scalar so JSON/orjson can encode it.
        # Use item() to convert numpy scalar -> Python native.
        return {**base, "encoding": "scalar", "value": a.flat[0].item()}

    layered = _sniff_layered(a)
    if layered is not None:
        axis, layer_values = layered
        return {
            **base,
            "encoding": "layered",
            "axis": axis,
            "data": _b64_encode(layer_values),
        }

    sparse = _sniff_sparse(
        a,
        min_sparse_cells=min_sparse_cells,
        sparse_density_threshold=sparse_density_threshold,
    )
    if sparse is not None:
        fill_value, indices, values = sparse
        return {
            **base,
            "encoding": "sparse",
            "fill": fill_value.item(),
            "indices": _b64_encode(indices),  # int32 flat indices
            "values": _b64_encode(values),
        }

    # Dense fallback
    return {**base, "encoding": "dense", "data": _b64_encode(a)}


def deserialize_ndarray(
    data: typing.Mapping[str, typing.Any] | typing.Sequence[typing.Any] | npt.NDArray,
    *,
    dtype: npt.DTypeLike = None,
) -> npt.NDArray:
    """
    Reconstructs a numpy array from a wire dict produced by `serialize_ndarray`.
    """
    if isinstance(data, np.ndarray):
        return data.astype(dtype, copy=False) if dtype is not None else data  # type: ignore[return-value]

    if isinstance(data, (list, tuple)):
        return np.asarray(data, dtype=dtype)

    if not isinstance(data, typing.Mapping):
        raise TypeError(f"Expected dict, list, or ndarray; got {type(data).__name__!r}")

    if not data.get("__ndarray__", None):  # type: ignore
        raise ValueError("Missing '__ndarray__' sentinel.")

    stored_dtype = np.dtype(data["dtype"])  # type: ignore
    shape = tuple(int(s) for s in data["shape"])  # type: ignore
    encoding = data.get("encoding", "dense")  # type: ignore # legacy dicts have no encoding key

    if encoding == "empty":
        array = np.empty(shape, dtype=stored_dtype)

    elif encoding == "scalar":
        array = np.full(shape, fill_value=data["value"], dtype=stored_dtype)  # type: ignore

    elif encoding == "layered":
        axis = int(data["axis"])  # type: ignore
        n_layers = shape[axis]
        layer_values = _b64_decode(data["data"], stored_dtype, (n_layers,))  # type: ignore
        array = np.empty(shape, dtype=stored_dtype)
        for idx, val in enumerate(layer_values):
            # Build index tuple: slice(None) for all axes except `axis`
            idx_tuple = tuple(idx if dim == axis else slice(None) for dim in range(array.ndim))
            array[idx_tuple] = val

    elif encoding == "sparse":
        fill_value = stored_dtype.type(data["fill"])  # type: ignore
        array = np.full(shape, fill_value=fill_value, dtype=stored_dtype)
        n_non_fill = int(
            len(base64.b64decode(data["indices"])) // np.dtype(np.int32).itemsize  # type: ignore
        )
        indices = _b64_decode(data["indices"], np.int32, (n_non_fill,))  # type: ignore
        values = _b64_decode(data["values"], stored_dtype, (n_non_fill,))  # type: ignore
        array.ravel()[indices] = values

    elif encoding == "dense":
        n_elements = int(np.prod(shape)) if shape else 1
        raw = base64.b64decode(data["data"])  # type: ignore
        expected = stored_dtype.itemsize * n_elements
        if len(raw) != expected:
            raise ValueError(f"Byte-length mismatch. Expected {expected}, got {len(raw)}")
        array = np.frombuffer(raw, dtype=stored_dtype).reshape(shape).copy()

    else:
        raise ValueError(f"Unknown encoding {encoding!r}")

    return array.astype(dtype, copy=False) if dtype is not None else array
