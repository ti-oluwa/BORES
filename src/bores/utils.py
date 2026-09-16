import base64
import logging
import math
import typing
from datetime import datetime, timedelta

import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import orjson
from numba.extending import overload  # type: ignore[import-untyped]

from bores.precision import get_dtype
from bores.types import (
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = ["array", "clip", "is_array"]


# NOTE: Future me! Do not import `bores.constant` at top level in this file. It leads to a circular import issue.


def array(obj: typing.Any, **kwargs: typing.Any):
    """
    Wrapper around `np.array` to enforce global dtype.

    :param obj: Object to convert to numpy array
    :param kwargs: Additional keyword arguments for `np.array`
    :return: return value of `np.array`
    """
    kwargs.setdefault("dtype", get_dtype())
    return np.array(obj, **kwargs)


@numba.vectorize(cache=True)
def clip(val, min_, max_):
    return np.maximum(np.minimum(val, max_), min_)


@numba.njit(cache=True)
def clip_scalar(value: float, min_val: float, max_val: float) -> float:
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


@typing.overload
def is_array(x: npt.NDArray) -> typing.TypeGuard[npt.NDArray]: ...


@typing.overload
def is_array(x: typing.Any) -> typing.TypeGuard[npt.NDArray]: ...


@numba.njit(cache=True)
def is_array(x: typing.Any) -> bool:
    return hasattr(x, "shape") and isinstance(x.shape, tuple)


@numba.njit(cache=True)
def _apply_mask_2d(a: npt.NDArray, mask: npt.NDArray, values: npt.NDArray) -> None:
    """
    Apply values (scalar or array) to a 2D array where mask is True (in-place).

    :param a: 2D array to modify
    :param mask: 2D boolean mask with same shape as a
    :param values: scalar or 2D array of values to assign where mask is True
    """
    nx, ny = a.shape

    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            if mask[i, j]:
                a[i, j] = values[i, j]


@numba.njit(cache=True)
def _apply_mask_3d(a: npt.NDArray, mask: npt.NDArray, values: npt.NDArray) -> None:
    """
    Apply values (scalar or array) to a 3D array where mask is True (in-place).

    :param a: 3D array to modify
    :param mask: 3D boolean mask with same shape as a
    :param values: scalar or 3D array of values to assign where mask is True
    """
    nx, ny, nz = a.shape
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    a[i, j, k] = values[i, j, k]


@numba.njit(cache=True)
def _apply_mask_nd(a: npt.NDArray, mask: npt.NDArray, values: npt.NDArray) -> None:
    """
    Apply values (scalar or array) to an N-dimensional array where mask is True (in-place).

    :param a: N-dimensional array to modify
    :param mask: N-dimensional boolean mask with same shape as a
    :param values: scalar or N-dimensional array of values to assign where mask is True
    """
    for idx in np.ndindex(a.shape):
        if mask[idx]:
            a[idx] = values[idx]


@numba.njit(cache=True)
def apply_mask(a: npt.NDArray, mask: npt.NDArray, values: npt.NDArray) -> None:
    """
    Dispatcher to apply scalar or array values to an array where mask is True.

    :param a: Array to modify (2D, 3D, or N-dimensional)
    :param mask: Boolean mask with same shape as a
    :param values: scalar or array of values to assign where mask is True
    """
    ndim = a.ndim
    if ndim == 2:
        _apply_mask_2d(a, mask, values)
    elif ndim == 3:
        _apply_mask_3d(a, mask, values)
    else:
        _apply_mask_nd(a, mask, values)


@numba.njit(cache=True)
def _get_mask_2d(a: npt.NDArray, mask: npt.NDArray, fill_value: float):
    """
    Return a new 2D array where values are kept if mask is True, otherwise replaced with fill_value.

    :param a: 2D input array
    :param mask: 2D boolean mask with same shape as a
    :param fill_value: Scalar value to fill where mask is False
    :return: 2D array with masked values applied
    """
    nx, ny = a.shape
    out = np.empty_like(a)
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            if mask[i, j]:
                out[i, j] = a[i, j]
            else:
                out[i, j] = fill_value
    return out


@numba.njit(cache=True)
def _get_mask_3d(a: npt.NDArray, mask: npt.NDArray, fill_value: float):
    """
    Return a new 3D array where values are kept if mask is True, otherwise replaced with fill_value.

    :param a: 3D input array
    :param mask: 3D boolean mask with same shape as a
    :param fill_value: Scalar value to fill where mask is False
    :return: 3D array with masked values applied
    """
    nx, ny, nz = a.shape
    out = np.empty_like(a)
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    out[i, j, k] = a[i, j, k]
                else:
                    out[i, j, k] = fill_value
    return out


@numba.njit(cache=True)
def _get_mask_nd(a: npt.NDArray, mask: npt.NDArray, fill_value: float):
    """
    Return a new N-dimensional array where values are kept if mask is True, otherwise replaced with fill_value.

    :param a: N-dimensional input array
    :param mask: N-dimensional boolean mask with same shape as a
    :param fill_value: Scalar value to fill where mask is False
    :return: N-dimensional array with masked values applied
    """
    out = np.empty_like(a)
    for idx in np.ndindex(a.shape):
        if mask[idx]:
            out[idx] = a[idx]
        else:
            out[idx] = fill_value
    return out


@numba.njit(cache=True)
def get_mask(a: npt.NDArray, mask: npt.NDArray, fill_value: float = np.nan):
    """
    Dispatcher to return a masked copy of an array.

    :param a: Input array (2D, 3D, or N-dimensional)
    :param mask: Boolean mask with same shape as a
    :param fill_value: Scalar value to fill where mask is False
    :return: Array with masked values applied
    """
    ndim = a.ndim
    if ndim == 2:
        return _get_mask_2d(a, mask, fill_value)
    elif ndim == 3:
        return _get_mask_3d(a, mask, fill_value)
    return _get_mask_nd(a, mask, fill_value)


# When used in pure-python, this called
def min_(x) -> np.floating[typing.Any]:
    if isinstance(x, float):
        return x  # type: ignore[return-value]
    return np.min(x)


def max_(x) -> np.floating[typing.Any]:
    if isinstance(x, float):
        return x  # type: ignore[return-value]
    return np.max(x)


# In numba context, these overloads are used
@overload(min_)
def min_overload(x):
    # SCALAR CASE
    if isinstance(x, numba.types.Number):

        def impl(x):
            return x

        return impl

    # ARRAY CASE
    if isinstance(x, numba.types.Array):

        def impl(x):
            return np.min(x)

        return impl


@overload(max_)
def max_overload(x):
    if isinstance(x, numba.types.Number):

        def impl(x):
            return x

        return impl

    if isinstance(x, numba.types.Array):

        def impl(x):
            return np.max(x)

        return impl


def atleast_1d(x, dtype: npt.DTypeLike = None) -> npt.NDArray:  # type: ignore
    return np.atleast_1d(x)


@overload(atleast_1d)
def overload_to_1d(x, dtype=None):

    if isinstance(dtype, numba.types.NoneType) or dtype is None:
        target_dtype = np.float64

    elif isinstance(dtype, numba.types.DTypeSpec):
        target_dtype = dtype.dtype

    else:
        return None  # no match

    # array case
    if isinstance(x, numba.types.Array):

        def impl(x, dtype=None):  # type: ignore
            return x

        return impl

    # scalar case
    if isinstance(x, (numba.types.Float, numba.types.Integer)):

        def impl(x, dtype=None):
            a = np.empty(1, dtype=target_dtype)
            a[0] = x
            return a

        return impl

    return None


def _numpy_default(obj: typing.Any) -> typing.Mapping[str, typing.Any]:
    if isinstance(obj, np.ndarray):
        # Small arrays are stored as JSON list
        if obj.ndim <= 2 and obj.size < 50:
            return {
                "__ndarray__": True,
                "encoding": "list",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist(),
            }

        # Large arrays are stored as base64
        return {
            "__ndarray__": True,
            "encoding": "base64",
            "dtype": str(obj.dtype),
            "shape": obj.shape,
            "data": base64.b64encode(obj.tobytes()).decode(),
        }

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _decode_ndarray(obj: typing.Mapping[str, typing.Any]) -> npt.NDArray:
    dtype = np.dtype(obj["dtype"])
    shape = tuple(obj["shape"])

    if obj["encoding"] == "list":
        return np.array(obj["data"], dtype=dtype).reshape(shape)

    if obj["encoding"] == "base64":
        raw = base64.b64decode(obj["data"])
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    raise TypeError(
        "Data for type `np.ndarray` is not deserializable. Data may have been corrupted"
    )


def _walk(obj: typing.Any) -> typing.Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return _decode_ndarray(obj)
        return {k: _walk(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_walk(v) for v in obj]

    return obj


def safe_json_dumps(data: typing.Any) -> bytes:
    return orjson.dumps(data, default=_numpy_default, option=orjson.OPT_INDENT_2)


def safe_json_loads(data: typing.Any) -> typing.Any:
    return _walk(orjson.loads(data))


def close_iterator(iter: typing.Any) -> None:
    """Helper to close an iterator if it has a close method."""
    close_method = getattr(iter, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception as exc:
            logger.warning(f"Error closing iterator: {exc}", exc_info=True)


def is_scalar_like(value):
    return np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0)


@typing.overload
def scale(a: NumberArray[NDimension], /, factor: Number) -> NumberArray[NDimension]: ...
@typing.overload
def scale(a: Number, /, factor: Number) -> Number: ...
@typing.overload
def scale(a: None, /, factor: Number) -> None: ...
def scale(
    a: NumberOrArray[NDimension] | None, /, factor: Number
) -> NumberOrArray[NDimension] | None:
    """Return `a * factor` as the same dtype; identity when factor == 1."""
    if a is None or factor == 1:
        return a

    result = a * factor
    if isinstance(a, np.ndarray):
        result = result.astype(a.dtype, copy=False)  # type: ignore[attr-defined]
    return typing.cast(NumberOrArray[NDimension], result)


@typing.overload
def scale_non_empty(a: NumberArray[NDimension], /, factor: Number) -> NumberArray[NDimension]: ...
@typing.overload
def scale_non_empty(a: Number, /, factor: Number) -> Number: ...
@typing.overload
def scale_non_empty(a: None, /, factor: Number) -> None: ...
def scale_non_empty(
    a: NumberOrArray[NDimension] | None, /, factor: Number
) -> NumberOrArray[NDimension] | None:
    """Scale only if the optional array or float is non-empty and truthy."""
    if isinstance(a, np.ndarray):
        non_empty = a is not None and a.size > 0
    else:
        non_empty = bool(a)
    return scale(a, factor) if non_empty else a


@typing.overload
def scale_and_offset(
    a: NumberArray[NDimension], /, factor: Number, offset: Number
) -> NumberArray[NDimension]: ...
@typing.overload
def scale_and_offset(a: Number, /, factor: Number, offset: Number) -> Number: ...
@typing.overload
def scale_and_offset(a: None, /, factor: Number, offset: Number) -> None: ...
def scale_and_offset(
    a: NumberOrArray[NDimension] | None, /, factor: Number, offset: Number
) -> NumberOrArray[NDimension] | None:
    """Return `a * factor + offset` as the same dtype; identity when trivial."""
    if a is None or (factor == 1 and offset == 0):
        return a

    result = (a * factor) + offset
    if isinstance(a, np.ndarray):
        result = result.astype(a.dtype, copy=False)  # type: ignore[attr-defined]
    return typing.cast(NumberOrArray[NDimension], result)


def get_hydrostatic_gradient_factor(unit_system: UnitSystem) -> float:
    """
    Return the hydrostatic pressure-gradient constant for the given unit
    system.

    The returned value is the multiplier used to convert density into the
    local pressure gradient in each unit system.
    """
    from bores.constants import c

    if unit_system == UnitSystem.FIELD:
        return c.HYDROSTATIC_GRADIENT_FACTOR_FIELD  # psi per (lbm/ft^3 * ft)
    elif unit_system == UnitSystem.METRIC:
        return c.HYDROSTATIC_GRADIENT_FACTOR_METRIC  # bar per (kg/m^3 * m)
    elif unit_system == UnitSystem.LAB:
        return c.HYDROSTATIC_GRADIENT_FACTOR_LAB  # atm per (g/cm^3 * cm)
    elif unit_system == UnitSystem.SI:
        return c.HYDROSTATIC_GRADIENT_FACTOR_SI  # Pa per (kg/m^3 * m)
    raise ValueError(f"Unsupported unit system: {unit_system!r}")


TIME_UNIT_PER_UNIT_SYSTEM = {
    UnitSystem.FIELD: "days",
    UnitSystem.METRIC: "days",
    UnitSystem.LAB: "hours",
    UnitSystem.SI: "seconds",
}
"""Elapsed-time unit each `UnitSystem` uses, following Eclipse/OPM convention."""


def get_current_date(
    *, start_date: datetime, elapsed_time: Number, unit_system: UnitSystem
) -> datetime:
    """
    Converts an elapsed-time offset into a calendar date.

    :param start_date: The run's start date.
    :param elapsed_time: Time since `start_date`, in `unit_system`'s own
        time unit (days for `FIELD`/`METRIC`, hours for `LAB`, seconds for `SI`).
    :param unit_system: Determines `elapsed_time`'s unit.
    :returns: `start_date` plus `elapsed_time`.
    """
    return start_date + timedelta(**{TIME_UNIT_PER_UNIT_SYSTEM[unit_system]: float(elapsed_time)})


def get_current_time(
    *, start_date: datetime, current_date: datetime, unit_system: UnitSystem
) -> float:
    """
    Converts a calendar date into an elapsed-time offset from `start_date`.

    :param start_date: The run's start date.
    :param current_date: The date to find the elapsed time to.
    :param unit_system: Determines the result's unit (days for
        `FIELD`/`METRIC`, hours for `LAB`, seconds for `SI`).
    :returns: `current_date - start_date`, in `unit_system`'s own time unit.
    """
    from bores.constants import c

    total_seconds = (current_date - start_date).total_seconds()
    if unit_system is UnitSystem.SI:
        return total_seconds
    if unit_system is UnitSystem.LAB:
        return total_seconds * c.DAYS_PER_SECOND * c.HOURS_PER_DAY
    return total_seconds * c.DAYS_PER_SECOND


def none_if_nan(value: Number) -> Number | None:
    """
    Returns `value`, or `None` if it's `NaN`.

    :param value: A possibly-`NaN` number read from a compiled array.
    :returns: `value`, or `None` if it's `NaN`.
    """
    return None if math.isnan(value) else value
