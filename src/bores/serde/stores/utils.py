"""Shared helpers for array-backed store backends (`hdf5.py`, `zarr.py`)."""

import typing
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "NONE_SENTINEL",
    "denormalize_from_storage",
    "is_none_sentinel",
    "normalize_for_storage",
    "normalize_loaded_value",
    "sequence_to_ndarray",
]

# Sentinel for None values to avoid object dtype issues
NONE_SENTINEL = "__NONE_SENTINEL__"

StringArrayFactory = typing.Callable[[typing.Sequence[str]], npt.NDArray]


def is_none_sentinel(value: typing.Any) -> bool:
    """Check if value is the None sentinel."""
    return isinstance(value, str) and value == NONE_SENTINEL


def sequence_to_ndarray(
    value: Sequence,
    path: str,
    string_array_factory: StringArrayFactory,
) -> npt.NDArray:
    """
    Convert a (possibly nested) sequence into a NumPy array safe for the
    calling backend's storage.

    Raises if the sequence would produce dtype=object for anything other
    than the string case (`string_array_factory` is the one sanctioned
    object-dtype exception, and only the caller's factory decides how that's
    represented on disk).

    :param value: Sequence to convert.
    :param path: Dotted/encoded path, for error messages only.
    :param string_array_factory: Builds the array for an all-`str` sequence.
        HDF5 and Zarr each have a different "correct" native representation
        for variable-length strings, so this is supplied by the caller
        rather than hardcoded here.
    """
    if not value:
        return np.empty((0,), dtype=np.int8)

    if any(isinstance(v, Mapping) for v in value):
        raise TypeError(f"Sequence of mappings must be stored as groups, not datasets: {path}")

    if isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes)):
        arrays = [sequence_to_ndarray(v, path, string_array_factory) for v in value]
        try:
            array = np.stack(arrays)
        except ValueError as exc:
            raise TypeError(f"Inconsistent nested sequence shapes at {path}") from exc
        return array

    if all(isinstance(v, (bool, np.bool_)) for v in value):
        return np.asarray(value, dtype=bool)

    if all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in value
    ):
        return np.asarray(value)

    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
        return np.asarray(value)

    if all(isinstance(v, str) for v in value):
        return string_array_factory(value)

    raise TypeError(
        f"Unsupported or mixed sequence contents at {path}: {set(type(v).__name__ for v in value)}"
    )


def normalize_for_storage(value: typing.Any) -> typing.Any:
    """Normalize Python values for storage (replace None with sentinel)."""
    if value is None:
        return NONE_SENTINEL
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        return [normalize_for_storage(v) for v in value]
    elif isinstance(value, Mapping):
        return {k: normalize_for_storage(v) for k, v in value.items()}
    return value


def denormalize_from_storage(value: typing.Any) -> typing.Any:
    """Denormalize values from storage (replace sentinel with None)."""
    if is_none_sentinel(value):
        return None
    elif isinstance(value, list):
        return [denormalize_from_storage(v) for v in value]
    elif isinstance(value, dict):
        return {k: denormalize_from_storage(v) for k, v in value.items()}
    return value


def normalize_loaded_value(value: typing.Any) -> typing.Any:
    """Normalize values loaded from an array-backed dataset."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S", "O", "T"):
            # "T" = numpy's variable-length StringDType (numpy >= 2.0),
            # native to zarr v3 although not present in older numpy, but the
            # `.kind` check is safe either way since older numpy simply
            # never produces it.
            result = value.astype(str).tolist()
            return denormalize_from_storage(result)
        return value
    return denormalize_from_storage(value)
