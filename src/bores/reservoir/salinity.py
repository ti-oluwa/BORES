"""Water salinity of the reservoir cells."""

import numpy as np
import numpy.typing as npt

from bores.correlations.water import compute_salinity_from_molality
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import CellArray, Integer, Number

__all__ = ["load_salinity", "resolve_salinity"]

MAXIMUM_SALINITY = 1.0e6
"""Salinity of pure salt, in ppm by weight, which no brine can exceed."""


def resolve_salinity(
    salinity: Number | CellArray,
    *,
    n_cells: Integer,
    dtype: npt.DTypeLike = None,  # type: ignore[assignment]
) -> CellArray:
    """
    Returns the salinity of every cell, from one value for the whole reservoir or from one per cell.

    Salinity is in ppm by weight of salt, in every unit system, since it is a ratio of masses.

    :param salinity: A single salinity for every cell, or the salinity of each cell.
    :param n_cells: Number of cells.
    :param dtype: Dtype of the result. `bores.precision.get_dtype()` if not given.
    :returns: The salinity of each cell, shape `(n_cells,)`.
    :raises ValidationError: If the salinity has the wrong shape, is not finite, or is outside
        the range from 0 to 1,000,000 ppm.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    values = np.asarray(salinity, dtype=dtype)
    if values.ndim == 0:
        values = np.full(n_cells, values, dtype=dtype)
    if values.shape != (n_cells,):
        raise ValidationError(
            f"`salinity` has shape {values.shape}; expected a scalar or ({n_cells},)."
        )
    if n_cells > 0 and (
        not np.isfinite(values).all() or values.min() < 0.0 or values.max() > MAXIMUM_SALINITY
    ):
        raise ValidationError(
            f"`salinity` must be between 0 and {MAXIMUM_SALINITY:g} ppm in every cell."
        )
    return values  # type: ignore[return-value]


def load_salinity(
    deck_file: DeckFile,
    *,
    n_cells: Integer,
    dtype: npt.DTypeLike = None,  # type: ignore[assignment]
) -> CellArray | None:
    """
    Returns the salinity of every cell, from the deck's `SALINITY` keyword.

    The deck gives the molality of NaCl in mol per kg of water, which is converted to ppm by
    weight of salt.

    :param deck_file: The deck to read.
    :param n_cells: Number of cells.
    :param dtype: Dtype of the result. `bores.precision.get_dtype()` if not given.
    :returns: The salinity of each cell, or `None` if the deck has no `SALINITY`.
    """
    record = deck_file.get("SALINITY")
    if record is None:
        return None
    return resolve_salinity(
        float(compute_salinity_from_molality(float(record["molality"]))),
        n_cells=n_cells,
        dtype=dtype,
    )
