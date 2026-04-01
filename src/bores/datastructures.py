import typing

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypedDict

from bores.errors import ValidationError
from bores.serialization import Serializable, deserialize_ndarray, serialize_ndarray
from bores.types import T
from bores.utils import clip

DType = typing.TypeVar("DType", float, np.floating)
ShapeT = typing.TypeVar("ShapeT", bound=typing.Tuple[int, ...])


__all__ = [
    "BottomHolePressure",
    "BottomHolePressure",
    "PhaseRange",
    "Range",
    "Rates",
    "SparseTensor",
]


class SparseTensor(Serializable, typing.Generic[DType, ShapeT]):
    """
    Sparse N-dimensional numeric tensor using a shape-typed key index.

    Stores only non-default entries internally, making it memory-efficient
    for large sparse N-dimensional arrays. Supports standard dictionary
    access, arithmetic operators, and NumPy interoperability.

    Generic over `DType` (the numeric scalar type of stored values) and
    `ShapeT` (the shape tuple, encoding rank and dimension sizes at the
    type level). Because `shape` is required at construction, all valid
    keys are exactly the set of N-tuples within the declared bounds -
    there is no separate `Key` type alias needed.

    Example:
    ```python
    # 2D sparse matrix
    m: SparseTensor[np.float64, Tuple[int, int]] = SparseTensor(
        shape=(3, 3), dtype=np.float64
    )
    m[0, 1] = 5.0

    # 3D sparse tensor
    t: SparseTensor[np.float32, Tuple[int, int, int]] = SparseTensor(
        shape=(4, 4, 4), dtype=np.float32
    )
    t[0, 1, 2] = 3.14

    arr = t.array()   # shape (4, 4, 4), dtype float32
    ```
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        shape: ShapeT,
        *,
        dtype: typing.Type[DType] = np.float64,
        default: typing.Union[float, np.floating] = 0.0,
        data: typing.Optional[
            typing.Union[
                typing.Mapping[ShapeT, DType],
                "SparseTensor[DType, ShapeT]",
            ]
        ] = None,
    ) -> None:
        """
        Initialize the sparse tensor.

        :param shape: Required N-dimensional shape tuple `(d0, d1, ..., dN)`,
            typed as `ShapeT`. Defines the rank and bounds of all valid keys.
            All keys are therefore of type `ShapeT`.
        :param dtype: NumPy scalar dtype for stored values and array conversion,
            typed as `DType`. Defaults to `np.float64`.
        :param default: The implicit fill value for missing keys. Entries set
            to this value are not stored. Defaults to `0.0`.
        :param data: Initial entries as a mapping from `ShapeT` keys to numeric
            values, or another `SparseTensor`. Applied after shape/dtype/default
            are set. Defaults to None.
        """
        self._shape: ShapeT = shape
        self._ndim: int = len(shape)
        self._dtype = dtype
        self._default: DType = dtype(default)  # type: ignore
        self._data: dict[ShapeT, DType] = {}

        if data is not None:
            self.update(data)

    def __getitem__(self, key: ShapeT) -> DType:
        """
        Retrieve the value at the given N-dimensional index, returning the
        default fill value if the entry is absent.

        :param key: An `ShapeT` integer tuple within the declared shape bounds.
        :returns: Stored value or the instance's `default`.
        :raises KeyError: If `key` is not a valid non-negative integer tuple
            of the correct rank.
        :raises IndexError: If `key` falls outside the declared `shape`.
        """
        self._validate_key(key)
        self._check_bounds(key)
        return self._data.get(key, self._default)

    def __setitem__(self, key: ShapeT, value: typing.Union[float, np.floating]) -> None:
        """
        Set the value at the given N-dimensional index.

        Entries equal to `default` are discarded to maintain sparsity.

        :param key: A `ShapeT` integer tuple within the declared shape bounds.
        :param value: Numeric value to store. Will be cast to `dtype`.
        :raises KeyError: If `key` is not a valid non-negative integer tuple
            of the correct rank.
        :raises IndexError: If `key` falls outside the declared `shape`.
        """
        self._validate_key(key)
        self._check_bounds(key)
        cast: DType = self._dtype(value)  # type: ignore
        if cast == self._default:
            self._data.pop(key, None)
        else:
            self._data[key] = cast

    def __delitem__(self, key: ShapeT) -> None:
        """
        Delete the explicitly stored entry at the given index.

        After deletion, accessing `key` returns `default` rather than
        raising `KeyError`.

        :param key: A `ShapeT` integer tuple.
        :raises KeyError: If `key` is not currently stored (already at default).
        """
        self._validate_key(key)
        del self._data[key]

    def __iter__(self) -> typing.Iterator[ShapeT]:
        """
        Iterate over all explicitly stored N-dimensional keys.

        Keys whose value equals `default` are never stored and will not
        appear in iteration.

        :returns: Iterator of `ShapeT` tuples.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """
        Return the number of explicitly stored (non-default) entries.

        :returns: Count of stored entries.
        """
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self._shape}, "
            f"dtype={self._dtype.__name__}, "
            f"default={self._default}, "
            f"nnz={self.nnz})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __contains__(self, key: object) -> bool:
        """
        Check whether a key has an explicitly stored (non-default) value.

        :param key: Key to test; must be a valid `ShapeT` integer tuple.
        :returns: True if the entry is stored, False otherwise.
        """
        if not isinstance(key, tuple):
            return False

        try:
            self._validate_key(key)  # type: ignore
        except KeyError:
            return False
        return key in self._data

    def __eq__(self, other: object) -> bool:
        """
        Test equality with another `SparseTensor`.

        Two instances are equal when their stored entries, shapes, dtypes,
        and defaults all match.

        :param other: Object to compare against.
        :returns: True if equal, False otherwise.
        """
        if not isinstance(other, SparseTensor):
            return False

        return (
            self._shape == other._shape
            and self._dtype == other._dtype
            and self._default == other._default
            and self._data == other._data
        )

    def __add__(self, other: "SparseTensor[DType, ShapeT]", /) -> Self:
        """
        Element-wise addition: `self + other`.

        :param other: Another `SparseTensor` of the same `DType` and `ShapeT`.
        :returns: New `SparseTensor` with summed entries.
        :raises TypeError: If `other` is not a `SparseTensor`.
        """
        if not isinstance(other, SparseTensor):
            return NotImplemented

        result = self.copy()
        for k, v in other._data.items():
            result[k] = result[k] + v
        return result

    def __sub__(self, other: "SparseTensor[DType, ShapeT]", /) -> Self:
        """
        Element-wise subtraction: `self - other`.

        :param other: Another `SparseTensor` of the same `DType` and `ShapeT`.
        :returns: New `SparseTensor` with differenced entries.
        :raises TypeError: If `other` is not a `SparseTensor`.
        """
        if not isinstance(other, SparseTensor):
            return NotImplemented

        result = self.copy()
        for k, v in other._data.items():
            result[k] = result[k] - v
        return result

    def __mul__(self, scalar: typing.Union[float, np.floating], /) -> Self:
        """
        Scalar multiplication: `self * scalar`.

        :param scalar: Numeric multiplier.
        :returns: New `SparseTensor` with all entries scaled.
        :raises TypeError: If `scalar` is not numeric.
        """
        if not isinstance(scalar, (float, np.floating)):
            return NotImplemented

        result: SparseTensor[DType, ShapeT] = SparseTensor(  # type: ignore
            shape=self._shape, dtype=self._dtype, default=self._default
        )
        for k, v in self._data.items():
            result[k] = v * scalar
        return result  # type: ignore[return-value]

    def __rmul__(self, scalar: typing.Union[float, np.floating], /) -> Self:
        """
        Right scalar multiplication: `scalar * self`.

        :param scalar: Numeric multiplier.
        :returns: New `SparseTensor` with all entries scaled.
        """
        return self.__mul__(scalar)

    def __matmul__(
        self, other: "SparseTensor[DType, ShapeT]"
    ) -> typing.Union[DType, "SparseTensor[DType, typing.Any]"]:
        """
        Sparse generalised matrix multiplication: `self @ other`.

        Supported cases mirror NumPy's `matmul` semantics:

        - **1D @ 1D**: inner (dot) product, returns a scalar of `DType`.
        - **1D @ 2D**: row-vector times matrix, returns a rank-1 `SparseTensor`.
        - **2D @ 1D**: matrix times column-vector, returns a rank-1 `SparseTensor`.
        - **2D @ 2D**: standard matrix product, returns a rank-2 `SparseTensor`.
        - **3D+**: batched contraction over the last axis of `self` and the
        second-to-last axis of `other`. Delegates to NumPy via `array()`
        since sparsity provides no meaningful benefit for batched operations,
        and returns a `SparseTensor` built from the dense result.

        :param other: A `SparseTensor` whose leading contraction axis matches
            `self`'s trailing axis.
        :returns: Scalar `DType` for 1D @ 1D, or a new `SparseTensor` otherwise.
        :raises TypeError: If `other` is not a `SparseTensor`.
        :raises ValueError: If the contraction axes are incompatible, or if
            either operand is rank-0.
        """
        if not isinstance(other, SparseTensor):
            return NotImplemented

        if self._ndim == 0 or other._ndim == 0:
            raise ValueError("matmul does not support rank-0 tensors")

        # 1D @ 1D - dot product → scalar
        if self._ndim == 1 and other._ndim == 1:
            if self._shape[0] != other._shape[0]:
                raise ValueError(
                    f"1D dot product requires equal lengths, "
                    f"got {self._shape[0]} and {other._shape[0]}"
                )
            acc: DType = self._dtype(0)  # type: ignore
            for (i,), v1 in self._data.items():
                if (i,) in other._data:
                    acc = self._dtype(acc + v1 * other._data[(i,)])  # type: ignore
            return acc

        # 1D @ 2D - (n,) @ (n, m) → (m,)
        if self._ndim == 1 and other._ndim == 2:
            n, m = other._shape
            if self._shape[0] != n:
                raise ValueError(
                    f"Shape mismatch for 1D @ 2D: ({self._shape[0]},) @ {other._shape}"
                )
            result: SparseTensor[DType, typing.Tuple[int]] = SparseTensor(  # type: ignore
                shape=(m,), dtype=self._dtype, default=self._default
            )
            for (i,), v1 in self._data.items():
                for (i2, j), v2 in other._data.items():
                    if i == i2:
                        result[(j,)] = result[(j,)] + v1 * v2  # type: ignore
            return result

        # 2D @ 1D - (n, m) @ (m,) → (n,)
        if self._ndim == 2 and other._ndim == 1:
            n, m = self._shape
            if m != other._shape[0]:
                raise ValueError(
                    f"Shape mismatch for 2D @ 1D: {self._shape} @ ({other._shape[0]},)"
                )
            result: SparseTensor[DType, typing.Tuple[int]] = SparseTensor(  # type: ignore
                shape=(n,), dtype=self._dtype, default=self._default
            )
            for (i, k), v1 in self._data.items():
                if (k,) in other._data:
                    result[(i,)] = result[(i,)] + v1 * other._data[(k,)]  # type: ignore
            return result

        # 2D @ 2D - standard matrix product
        if self._ndim == 2 and other._ndim == 2:
            n, k = self._shape
            k2, m = other._shape
            if k != k2:
                raise ValueError(
                    f"Shape mismatch for 2D @ 2D: {self._shape} @ {other._shape}"
                )
            result: SparseTensor[DType, typing.Tuple[int, int]] = SparseTensor(  # type: ignore
                shape=(n, m), dtype=self._dtype, default=self._default
            )
            for (i, k1), v1 in self._data.items():
                for (k2, j), v2 in other._data.items():
                    if k1 == k2:
                        result[(i, j)] = result[(i, j)] + v1 * v2  # type: ignore
            return result

        # ND @ ND - batched contraction, delegate to NumPy
        if self._shape[-1] != other._shape[-2]:
            raise ValueError(
                f"Shape mismatch for batched matmul: "
                f"{self._shape} @ {other._shape} - "
                f"last axis of left ({self._shape[-1]}) must match "
                f"second-to-last of right ({other._shape[-2]})"
            )
        dense_result = np.matmul(self.array(), other.array())
        out_shape: typing.Tuple[int, ...] = dense_result.shape
        result: SparseTensor[DType, typing.Tuple[int, ...]] = SparseTensor(  # type: ignore
            shape=out_shape, dtype=self._dtype, default=self._default
        )
        for idx in zip(*np.nonzero(dense_result != self._default)):
            key = tuple(int(i) for i in idx)
            result[key] = self._dtype(dense_result[key])  # type: ignore
        return result

    def __truediv__(self, scalar: typing.Union[float, np.floating], /) -> Self:
        """
        Scalar division: `self / scalar`.

        :param scalar: Numeric divisor.
        :returns: New `SparseTensor` with all entries divided by `scalar`.
        :raises TypeError: If `scalar` is not numeric.
        :raises ZeroDivisionError: If `scalar` is zero.
        """
        if not isinstance(scalar, (float, np.floating)):
            return NotImplemented

        if scalar == 0:
            raise ZeroDivisionError("Sparse tensor division by zero")

        result: SparseTensor[DType, ShapeT] = SparseTensor(  # type: ignore
            shape=self._shape, dtype=self._dtype, default=self._default
        )
        for k, v in self._data.items():
            result[k] = v / scalar
        return result  # type: ignore[return-value]

    def __iadd__(self, other: "SparseTensor[DType, ShapeT]", /) -> Self:
        """
        In-place element-wise addition: `self += other`.

        Mutates `self` directly without allocating a new tensor.

        :param other: Another `SparseTensor` of the same `DType` and `ShapeT`.
        :returns: `self` after mutation.
        :raises TypeError: If `other` is not a `SparseTensor`.
        """
        if not isinstance(other, SparseTensor):
            return NotImplemented

        for k, v in other._data.items():
            self[k] = self[k] + v
        return self

    def __isub__(self, other: "SparseTensor[DType, ShapeT]", /) -> Self:
        """
        In-place element-wise subtraction: `self -= other`.

        Mutates `self` directly without allocating a new tensor.

        :param other: Another `SparseTensor` of the same `DType` and `ShapeT`.
        :returns: `self` after mutation.
        :raises TypeError: If `other` is not a `SparseTensor`.
        """
        if not isinstance(other, SparseTensor):
            return NotImplemented

        for k, v in other._data.items():
            self[k] = self[k] - v
        return self

    def __imul__(self, scalar: typing.Union[float, np.floating], /) -> Self:
        """
        In-place scalar multiplication: `self *= scalar`.

        Mutates `self` directly without allocating a new tensor.

        :param scalar: Numeric multiplier.
        :returns: `self` after mutation.
        :raises TypeError: If `scalar` is not numeric.
        """
        if not isinstance(scalar, (int, float, np.floating)):
            return NotImplemented

        for k in list(self._data):
            self[k] = self._data[k] * scalar
        return self

    def __itruediv__(self, scalar: typing.Union[float, np.floating], /) -> Self:
        """
        In-place scalar division: `self /= scalar`.

        Mutates `self` directly without allocating a new tensor.

        :param scalar: Numeric divisor.
        :returns: `self` after mutation.
        :raises TypeError: If `scalar` is not numeric.
        :raises ZeroDivisionError: If `scalar` is zero.
        """
        if not isinstance(scalar, (float, np.floating)):
            return NotImplemented

        if scalar == 0:
            raise ZeroDivisionError("Sparse tensor in-place division by zero")
        for k in list(self._data):
            self[k] = self._data[k] / scalar
        return self

    def __ipow__(self, exponent: typing.Union[float, np.floating], /) -> Self:
        """
        In-place element-wise power: `self **= exponent`.

        Only stored (non-default) entries are raised to the power. If `default`
        is `0.0` this is correct since `0 ** n == 0` for all positive `n`.

        :param exponent: Numeric exponent.
        :returns: `self` after mutation.
        :raises TypeError: If `exponent` is not numeric.
        """
        if not isinstance(exponent, (float, np.floating)):
            return NotImplemented

        for k in list(self._data):
            self[k] = self._data[k] ** exponent
        return self

    def __invert__(self) -> Self:
        """
        Element-wise multiplicative inverse: `~self`.

        Returns a new tensor where every stored entry `v` becomes `1 / v`.
        Absent (default) entries are left as-is; inverting a zero-fill tensor
        element-wise is undefined for the missing entries and they remain zero.

        :returns: New `SparseTensor` with stored entries inverted.
        :raises ZeroDivisionError: If any stored entry is zero.
        """
        result: SparseTensor[DType, ShapeT] = SparseTensor(  # type: ignore
            shape=self._shape, dtype=self._dtype, default=self._default
        )
        for k, v in self._data.items():
            if v == 0:
                raise ZeroDivisionError(f"Cannot invert zero entry at {k}")
            result[k] = self._dtype(1) / v  # type: ignore
        return result  # type: ignore[return-value]

    def __neg__(self) -> Self:
        """
        Negate all stored entries: `-self`.

        :returns: New `SparseTensor` with all values negated.
        """
        result: SparseTensor[DType, ShapeT] = SparseTensor(  # type: ignore
            shape=self._shape, dtype=self._dtype, default=self._default
        )
        for k, v in self._data.items():
            result[k] = -v
        return result  # type: ignore[return-value]

    def __abs__(self) -> Self:
        """
        Element-wise absolute value: `abs(self)`.

        Returns a new tensor where each stored entry `v` becomes `|v|`.
        Entries equal to `default` remain implicit and are not stored.

        :returns: New `SparseTensor` with absolute-valued entries.
        """
        result: SparseTensor[DType, ShapeT] = SparseTensor(  # type: ignore
            shape=self._shape,
            dtype=self._dtype,
            default=self._default,
        )

        for k, v in self._data.items():
            abs_v = self._dtype(abs(v))  # type: ignore
            if abs_v != self._default:
                result[k] = abs_v

        return result  # type: ignore[return-value]

    def __array__(self, dtype: typing.Optional[npt.DTypeLike] = None):
        return self.array(dtype=dtype)

    @property
    def shape(self) -> ShapeT:
        """
        The N-dimensional shape of this tensor.

        :returns: Shape tuple `(d0, d1, ..., dN)` typed as `ShapeT`.
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions (rank) of this tensor.

        :returns: Number of dimensions.
        """
        return self._ndim

    @property
    def dtype(self) -> typing.Type[DType]:
        """
        The numeric scalar type used for stored values.

        :returns: A type such as `float` or `np.float64`.
        """
        return self._dtype

    @property
    def default(self) -> DType:
        """
        The implicit fill value returned for missing keys.

        :returns: The default value cast to `dtype`.
        """
        return self._default

    @property
    def nnz(self) -> int:
        """
        Number of explicitly stored (non-default) entries.

        :returns: Non-zero (or non-default) entry count.
        """
        return len(self._data)

    @property
    def empty(self) -> bool:
        """
        Whether this tensor has no explicitly stored entries.

        :returns: True if no entries are stored, False otherwise.
        """
        return len(self._data) == 0

    def copy(self) -> Self:
        """
        Return a shallow copy of this `SparseTensor`.

        Shape, dtype, and default are preserved.

        :returns: New `SparseTensor[DType, ShapeT]` with identical entries
            and metadata.
        """
        return SparseTensor(  # type: ignore
            shape=self._shape,
            dtype=self._dtype,
            default=self._default,
            data=self._data.copy(),
        )

    def clear(self) -> None:
        """
        Remove all stored entries, resetting the tensor to all-default.

        Shape, dtype, and default are retained.
        """
        self._data.clear()

    def get(
        self,
        key: ShapeT,
        default: typing.Optional[typing.Union[float, np.floating]] = None,
    ) -> DType:
        """
        Retrieve the value at the given index, returning a specified default if
        the entry is absent.

        :param key: An `ShapeT` integer tuple within the declared shape bounds.
        :param default: Optional override for the default fill value. If None,
            uses the instance's `default`.
        :returns: Stored value or the specified default.
        :raises KeyError: If `key` is not a valid non-negative integer tuple
            of the correct rank.
        :raises IndexError: If `key` falls outside the declared `shape`.
        """
        self._validate_key(key)
        self._check_bounds(key)
        if default is None:
            return self._data.get(key, self._default)

        cast_default: DType = self._dtype(default)  # type: ignore
        return self._data.get(key, cast_default)

    def update(
        self,
        map: typing.Union[
            typing.Mapping[ShapeT, DType], "SparseTensor[typing.Any, ShapeT]"
        ],
        /,
    ) -> None:
        for k in map:
            self[k] = map[k]

    def dense(self) -> typing.List[typing.Any]:
        """
        Convert to a dense N-dimensional Python nested list.

        Shape is always taken from the instance's declared `shape` - no
        override is needed since `shape` is required at construction.

        :returns: Nested list of values, filling absent entries with `default`.

        Example:
        ```python
        t: SparseTensor[np.float64, Tuple[int, int, int]] = SparseTensor(
            shape=(2, 2, 2), dtype=np.float64
        )
        t[(0, 1, 1)] = 9.0
        t.dense()
        # [[[0.0, 0.0], [0.0, 9.0]], [[0.0, 0.0], [0.0, 0.0]]]
        ```
        """

        def _build(dim: int, prefix: ShapeT) -> list:
            if dim == self._ndim:
                return self._data.get(prefix, self._default)  # type: ignore[return-value]
            return [_build(dim + 1, prefix + (i,)) for i in range(self._shape[dim])]  # type: ignore[return-value]

        return _build(0, ())  # type: ignore

    def array(
        self,
        dtype: typing.Optional[npt.DTypeLike] = None,
    ) -> npt.NDArray:
        """
        Convert to a dense NumPy N-dimensional array.

        Shape is always taken from the instance's declared `shape` - no
        override is needed since `shape` is required at construction.

        :param dtype: NumPy dtype override for the output array. Defaults to
            the instance's `dtype`.
        :returns: N-dimensional `np.ndarray` filled with stored values and
            `default` elsewhere, with shape equal to `self.shape`.

        Example:
        ```python
        t: SparseTensor[np.float32, Tuple[int, int, int]] = SparseTensor(
            shape=(2, 3, 4), dtype=np.float32
        )
        t[(0, 1, 2)] = 7.0
        arr = t.array()   # shape (2, 3, 4), dtype float32
        ```
        """
        out_dtype = dtype or self._dtype
        arr = np.full(self._shape, self._default, dtype=out_dtype)
        for key, v in self._data.items():
            arr[key] = v
        return arr

    def sum(self) -> DType:
        """
        Sum of all elements (including implicit default entries).
        """
        acc: DType = self._dtype(0)  # type: ignore
        for v in self._data.values():
            acc = self._dtype(acc + v)  # type: ignore

        total_size = int(np.prod(self._shape))
        missing = total_size - len(self._data)

        if self._default != 0:
            acc = self._dtype(acc + missing * self._default)  # type: ignore
        return acc

    def nansum(self) -> DType:
        """
        Sum ignoring NaNs (including implicit default entries).
        """
        acc: DType = self._dtype(0)  # type: ignore
        for v in self._data.values():
            if not np.isnan(v):
                acc = self._dtype(acc + v)  # type: ignore

        total_size = int(np.prod(self._shape))
        missing = total_size - len(self._data)

        if not np.isnan(self._default):
            acc = self._dtype(acc + missing * self._default)  # type: ignore
        return acc

    def mean(self) -> float:
        """
        Mean of all elements.
        """
        total_size = int(np.prod(self._shape))
        if total_size == 0:
            raise ZeroDivisionError("Mean of empty tensor")
        return float(self.sum()) / total_size

    def nanmean(self) -> float:
        """
        Mean ignoring NaNs.
        """
        total_size = int(np.prod(self._shape))

        count = 0
        acc: float = 0.0

        for v in self._data.values():
            if not np.isnan(v):
                acc += float(v)
                count += 1

        missing = total_size - len(self._data)

        if not np.isnan(self._default):
            acc += missing * float(self._default)
            count += missing

        if count == 0:
            return float("nan")

        return acc / count

    def min(self) -> DType:
        """
        Minimum value over all elements.
        """
        values = list(self._data.values())

        if len(self._data) < int(np.prod(self._shape)):
            values.append(self._default)

        if not values:
            raise ValueError("min() of empty tensor")

        return self._dtype(min(values))  # type: ignore

    def max(self) -> DType:
        """
        Maximum value over all elements.
        """
        values = list(self._data.values())

        if len(self._data) < int(np.prod(self._shape)):
            values.append(self._default)

        if not values:
            raise ValueError("max() of empty tensor")

        return self._dtype(max(values))  # type: ignore

    def nanmin(self) -> DType:
        """
        Minimum ignoring NaNs.
        """
        values = [v for v in self._data.values() if not np.isnan(v)]

        if len(self._data) < int(np.prod(self._shape)) and not np.isnan(self._default):
            values.append(self._default)

        if not values:
            return self._dtype(np.nan)  # type: ignore

        return self._dtype(min(values))  # type: ignore

    def nanmax(self) -> DType:
        """
        Maximum ignoring NaNs.
        """
        values = [v for v in self._data.values() if not np.isnan(v)]

        if len(self._data) < int(np.prod(self._shape)) and not np.isnan(self._default):
            values.append(self._default)

        if not values:
            return self._dtype(np.nan)  # type: ignore

        return self._dtype(max(values))  # type: ignore

    def abs(self) -> Self:
        """
        Element-wise absolute value.

        Returns a new tensor where each stored entry `v` becomes `|v|`.
        Entries equal to `default` remain implicit and are not stored.

        :returns: New `SparseTensor` with absolute-valued entries.
        """
        return self.__abs__()

    def _validate_key(self, key: typing.Any) -> None:
        """
        Assert that `key` is a valid non-negative integer tuple of rank
        matching `self.ndim`.

        :param key: The key to validate.
        :raises KeyError: If `key` is not a tuple of non-negative integers,
            or if its length does not match `ndim`.
        """
        if not isinstance(key, tuple) or not all(
            isinstance(k, int) and k >= 0 for k in key
        ):
            raise KeyError(f"Key must be a tuple of non-negative integers, got {key!r}")
        if len(key) != self._ndim:
            raise KeyError(
                f"Key rank {len(key)} does not match tensor rank {self._ndim}"
            )

    def _check_bounds(self, key: ShapeT) -> None:
        """
        Assert that `key` lies within the declared `shape`.

        :param key: A pre-validated `ShapeT` index tuple.
        :raises IndexError: If any component of `key` meets or exceeds the
            corresponding dimension in `self.shape`.
        """
        for idx, dim in zip(key, self._shape):
            if idx >= dim:
                raise IndexError(
                    f"Index {key} is out of bounds for shape {self._shape}"
                )

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """
        Serialize the sparse tensor to a dictionary.

        Stores only the non-default entries (_data) as parallel lists of
        keys and values to avoid per-entry dict overhead and keeping the
        wire format compact. Keys are flattened to lists (JSON-safe), values
        are stored as a raw base64 numpy array for exact dtype round-trip.
        """
        keys_flat: typing.List[typing.List[int]] = [list(k) for k in self._data]
        values_arr = (
            np.array(list(self._data.values()), dtype=self._dtype)
            if self._data
            else np.array([], dtype=self._dtype)
        )
        return {
            "shape": list(self._shape),
            "dtype": np.dtype(self._dtype).str,  # e.g. "<f8" - exact, portable
            "default": self._default,  # Python scalar, JSON-safe
            "keys": keys_flat,  # List[List[int]]
            "values": serialize_ndarray(values_arr),  # compact base64 blob
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """
        Deserialize a sparse tensor from a dictionary produced by `__dump__`.
        """
        shape = tuple(int(d) for d in data["shape"])
        dtype = np.dtype(data["dtype"]).type  # e.g. np.float64
        default = data["default"]

        values_arr = deserialize_ndarray(data["values"])
        keys_raw: typing.List[typing.Tuple[int, ...]] = [
            tuple(map(int, k)) for k in data["keys"]
        ]

        sparse_data: typing.Dict[typing.Tuple[int, ...], typing.Any] = {
            k: dtype(v) for k, v in zip(keys_raw, values_arr)
        }
        return cls(shape=shape, dtype=dtype, default=default, data=sparse_data)  # type: ignore[arg-type]


class BottomHolePressure(SparseTensor[DType, ShapeT]):
    """
    Sparse tensor of Bottom Hole Pressures with domain validation.
    """

    __abstract_serializable__ = True

    def __setitem__(self, key: ShapeT, value: typing.Union[float, np.floating]) -> None:
        if value < 0:
            raise ValueError(f"BHP must be non-negative, got {value} at {key}")
        super().__setitem__(key, value)


@attrs.frozen(slots=True)
class Rates(Serializable, typing.Generic[DType, ShapeT]):
    """
    Wrapper for N-dimensional sparse tensors representing phase flow rates
    (oil, water, gas).
    """

    oil: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil flow rates."""

    water: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil flow rates."""

    gas: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil flow rates."""

    @property
    def total(self) -> SparseTensor[DType, ShapeT]:
        """
        Compute the total fluid flow rate (oil + water + gas) at each cell.

        All defined phase tensors must share the same `ShapeT` and `DType`.
        If none of the phase tensors are defined, returns None.

        :returns: A new `SparseTensor[DType, ShapeT]` of summed rates, or
            None if all phases are absent.
        """
        return self.oil + self.water + self.gas

    def __iter__(self) -> typing.Iterator[typing.Optional[SparseTensor[DType, ShapeT]]]:
        yield self.water
        yield self.oil
        yield self.gas

    def __getitem__(self, key: ShapeT) -> typing.Tuple[DType, DType, DType]:
        """
        Return the water, oil, and gas flow rates at the specified cell.

        If a phase tensor is not defined, its rate is returned as the
        default fill value of the other defined tensors, falling back to
        `0.0` if none are defined.

        :param key: The N-dimensional cell index typed as `ShapeT`.
        :returns: A tuple `(water_rate, oil_rate, gas_rate)` each of type
            `DType`.
        """
        return self.water[key], self.oil[key], self.gas[key]

    def abs(self) -> Self:
        return Rates(  # type: ignore[return-value]
            oil=abs(self.oil),
            water=abs(self.water),
            gas=abs(self.gas),
        )


@attrs.frozen(slots=True)
class BottomHolePressures(Serializable, typing.Generic[DType, ShapeT]):
    """
    Wrapper for N-dimensional sparse tensors representing phase injection/production bottom hole pressures
    (oil, water, gas).
    """

    oil: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil bottom hole pressures."""

    water: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil bottom hole pressures."""

    gas: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil bottom hole pressures."""

    def __iter__(self) -> typing.Iterator[typing.Optional[SparseTensor[DType, ShapeT]]]:
        yield self.water
        yield self.oil
        yield self.gas

    def __getitem__(self, key: ShapeT) -> typing.Tuple[DType, DType, DType]:
        """
        Return the water, oil and gas bottom hole pressures at the specified cell.

        If a phase tensor is not defined, its bhp is returned as the
        default fill value of the other defined tensors, falling back to
        `0.0` if none are defined.

        :param key: The N-dimensional cell index typed as `ShapeT`.
        :returns: A tuple `(water_bhp, oil_bhp, gas_bhp)` each of type
            `DType`.
        """
        return self.water[key], self.oil[key], self.gas[key]

    def __setitem__(
        self,
        key: ShapeT,
        value: typing.Tuple[
            typing.Union[DType, float],
            typing.Union[DType, float],
            typing.Union[DType, float],
        ],
    ) -> None:
        """
        Set the water, oil and gas bottom hole pressures at the specified cell.

        :param key: The N-dimensional cell index typed as `ShapeT`.
        :param value: A three-tuple `(water_bhp, oil_bhp, gas_bhp)`.
        """
        water, oil, gas = value
        self.oil[key] = oil
        self.water[key] = water
        self.gas[key] = gas


@attrs.frozen(slots=True)
class FormationVolumeFactors(Serializable, typing.Generic[DType, ShapeT]):
    """
    Wrapper for N-dimensional sparse tensors representing phase formation volume factors
    (oil, water, gas).
    """

    oil: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil formation volume factors."""

    water: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil formation volume factors."""

    gas: SparseTensor[DType, ShapeT]
    """Sparse tensor representing oil formation volume factors."""

    def __iter__(self) -> typing.Iterator[typing.Optional[SparseTensor[DType, ShapeT]]]:
        yield self.water
        yield self.oil
        yield self.gas

    def __getitem__(self, key: ShapeT) -> typing.Tuple[DType, DType, DType]:
        """
        Return the water, oil and gas formation volume factors at the specified cell.

        If a phase tensor is not defined, its formation volume factor is returned as the
        default fill value of the other defined tensors, falling back to
        `0.0` if none are defined.

        :param key: The N-dimensional cell index typed as `ShapeT`.
        :returns: A tuple `(water_bhp, oil_bhp, gas_bhp)` each of type
            `DType`.
        """
        return self.water[key], self.oil[key], self.gas[key]

    def __setitem__(
        self,
        key: ShapeT,
        value: typing.Tuple[
            typing.Union[DType, float],
            typing.Union[DType, float],
            typing.Union[DType, float],
        ],
    ) -> None:
        """
        Set the water, oil and gas formation volume factors at the specified cell.

        :param key: The N-dimensional cell index typed as `ShapeT`.
        :param value: A three-tuple `(water_bhp, oil_bhp, gas_bhp)`.
        """
        water, oil, gas = value
        self.oil[key] = oil
        self.water[key] = water
        self.gas[key] = gas


@attrs.frozen(slots=True)
class PhaseTensorsProxy(typing.Generic[DType, ShapeT]):
    """
    Proxy allowing controlled item assignment on phase sparse tensors without
    exposing the underlying tensors directly.

    All three phase tensors must be fully initialised before constructing
    this proxy as it does not handle absent phases.

    :param oil: Sparse tensor for oil.
    :param water: Sparse tensor for water.
    :param gas: Sparse tensor for gas.
    """

    oil: SparseTensor[DType, ShapeT]
    water: SparseTensor[DType, ShapeT]
    gas: SparseTensor[DType, ShapeT]

    def __setitem__(
        self,
        key: ShapeT,
        value: typing.Tuple[
            typing.Union[DType, float],
            typing.Union[DType, float],
            typing.Union[DType, float],
        ],
    ) -> None:
        """
        Set the oil, water, and gas values at the specified cell.

        :param key: The N-dimensional cell index typed as `ShapeT`.
        :param value: A three-tuple `(oil_value, water_value, gas_value)`.
        """
        water, oil, gas = value
        self.oil[key] = oil
        self.water[key] = water
        self.gas[key] = gas

    def __getitem__(self, key: ShapeT) -> typing.Tuple[DType, DType, DType]:
        """
        Return the oil, water, and gas values at the specified cell.

        :param key: The N-dimensional cell index typed as `ShapeT`.
        :returns: A tuple `(oil_value, water_value, gas_value)` each of type
            `DType`.
        """
        return self.water[key], self.oil[key], self.gas[key]


@attrs.frozen
class Range(Serializable):
    """
    Class representing minimum and maximum values.
    """

    min: float
    """Minimum value."""
    max: float
    """Maximum value."""

    def __attrs_post_init__(self) -> None:
        if self.min > self.max:
            raise ValidationError("Minimum value cannot be greater than maximum value.")

    def clip(self, value: T) -> T:
        """
        Clips the given value between the minimum and maximum values.

        :param value: The value to be clipped.
        :return: The clipped value.
        """
        return clip(value, self.min, self.max)

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> typing.Iterator[float]:
        yield self.min
        yield self.max

    def __contains__(self, item: float) -> bool:
        return self.min <= item <= self.max

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.min
        elif index == 1:
            return self.max
        else:
            raise IndexError("Index out of range for Range. Valid indices are 0 and 1.")


class PhaseRange(TypedDict):
    """
    Dictionary holding ranges for different phases.
    """

    oil: Range
    water: Range
    gas: Range
