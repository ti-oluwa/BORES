"""Reusable workspace for one simulation run."""

import typing

import numpy as np
import numpy.typing as npt

from bores.precision import get_dtype
from bores.reservoir.state import ReservoirState
from bores.types import NumberArray, OneDimension

__all__ = [
    "ReservoirWorkspace",
    "build_reservoir_workspace",
    "load_reservoir_state",
]


class ReservoirWorkspace(typing.NamedTuple):
    """
    Per-cell reservoir primary unknowns, one row per cell, for the whole run.

    The reservoir state (`ReservoirState`) is decompiled from whatever this holds at a
    reported time.
    """

    pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Oil-phase reference pressure."""

    oil_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    water_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    gas_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    solution_gor: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rs."""

    oil_bubble_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    vaporized_oil_to_gas_ratio: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rv."""

    gas_dew_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    gas_solubility_in_water: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rsw."""

    water_bubble_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""


def build_reservoir_workspace(
    *, state: ReservoirState, dtype: npt.DTypeLike = None
) -> ReservoirWorkspace:
    """
    Builds a `ReservoirWorkspace` from a reservoir state's own primary unknowns.

    :param state: The reservoir state to seed the workspace from.
    :param n_cells: Number of grid cells.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `ReservoirWorkspace` seeded from `state`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    return ReservoirWorkspace(
        pressure=state.pressure.astype(dtype, copy=True),
        oil_saturation=state.oil_saturation.astype(dtype, copy=True),
        water_saturation=state.water_saturation.astype(dtype, copy=True),
        gas_saturation=state.gas_saturation.astype(dtype, copy=True),
        solution_gor=state.solution_gor.astype(dtype, copy=True),
        oil_bubble_point_pressure=state.oil_bubble_point_pressure.astype(dtype, copy=True),
        vaporized_oil_to_gas_ratio=state.vaporized_oil_to_gas_ratio.astype(dtype, copy=True),
        gas_dew_point_pressure=state.gas_dew_point_pressure.astype(dtype, copy=True),
        gas_solubility_in_water=state.gas_solubility_in_water.astype(dtype, copy=True),
        water_bubble_point_pressure=state.water_bubble_point_pressure.astype(dtype, copy=True),
    )


def load_reservoir_state(workspace: ReservoirWorkspace) -> ReservoirState: ...
