from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import numpy.typing as npt

__all__ = [
    "get_dtype",
    "get_floating_point_info",
    "set_dtype",
    "use_32bit_precision",
    "use_64bit_precision",
    "with_precision",
]

_bores_dtype: ContextVar[npt.DTypeLike] = ContextVar("_bores_dtype", default=np.float32)


def get_dtype() -> npt.DTypeLike:
    """
    Get the current data precision type for used for computations in `BORES`.

    This defines the precision used in calculations.

    :return: The current data type.
    """
    return _bores_dtype.get()


get_precision = get_dtype


def set_dtype(dtype: npt.DTypeLike) -> None:
    """
    Set the default data precision type for `BORES` computations.

    Useful for setting precision for current context.

    :param dtype: The data type to set as default.
    """
    _bores_dtype.set(dtype)


set_precision = set_dtype


@contextmanager
def with_precision(dtype: npt.DTypeLike):
    """
    Context manager to temporarily set the data type, and hence the precision for `BORES` computations.

    :param dtype: The data type to set within the context.
    """
    token = _bores_dtype.set(dtype)
    try:
        yield
    finally:
        _bores_dtype.reset(token)


def use_64bit_precision() -> None:
    """
    Set the default data precision type to float64 for `BORES` computations.
    """
    set_dtype(np.float64)


def use_32bit_precision() -> None:
    """
    Set the default data precision type to float32 for `BORES` computations.

    Default precision for `BORES`.
    """
    set_dtype(np.float32)


def get_floating_point_info() -> np.finfo[np.floating]:
    """
    Get the floating point information for the current data type used in `BORES` computations.

    :return: The floating point information.
    """
    dtype = get_dtype()
    return np.finfo(dtype)  # type: ignore
