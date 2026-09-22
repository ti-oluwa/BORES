"""Mutable, per-run aquifer state."""

import typing

import numpy as np
import numpy.typing as npt

from bores.precision import get_dtype
from bores.reservoir.boundary.compile import CompiledAquifers
from bores.types import NumberArray, OneDimension

__all__ = ["AquiferWorkspace", "build_aquifer_workspace"]


class AquiferWorkspace(typing.NamedTuple):
    """
    Every aquifer's own evolving recursive state, one row per aquifer in
    `CompiledAquifers` order. Only the fields a row's own kind actually
    uses are load-bearing; the rest sit at `0.0`.

    Every field here is the state as of the last committed timestep,
    read (but never written) by a trial evaluation mid-iteration, and
    advanced in place, once, on acceptance (`commit_boundary_conditions`).
    """

    previous_time: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. Simulation time as of the last committed state."""

    previous_cumulative_influx: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. `We` as of the last committed state. Both kinds."""

    previous_boundary_pressure: NumberArray[OneDimension]
    """
    Shape `(n_aquifers,)`. Average boundary pressure as of the last
    committed state. Kept for reporting/decompile; not read by either
    recurrence (each reads the current trial/accepted pressure directly).
    """

    previous_dimensionless_time: NumberArray[OneDimension]
    """Shape `(n_aquifers,)`. `tD` as of the last committed state. Carter-Tracy rows only."""

    previous_aquifer_pressure: NumberArray[OneDimension]
    """
    Shape `(n_aquifers,)`. The aquifer's own declining average pressure
    as of the last committed state. Fetkovich rows only, initialised to
    that row's own `initial_pressure`.
    """


def build_aquifer_workspace(
    compiled: CompiledAquifers, dtype: npt.DTypeLike = None
) -> AquiferWorkspace:
    """
    Builds a fresh `AquiferWorkspace` for `compiled`, every row at its
    initial (zero-influx, initial-pressure) state.

    :param compiled: The aquifer system to build a workspace for.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `AquiferWorkspace` with every aquifer at its starting state.
    """
    resolved_dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    n = compiled.kinds.shape[0]
    return AquiferWorkspace(
        previous_time=np.zeros(n, dtype=resolved_dtype),
        previous_cumulative_influx=np.zeros(n, dtype=resolved_dtype),
        previous_boundary_pressure=typing.cast(
            NumberArray[OneDimension],
            np.asarray(compiled.initial_pressures, dtype=resolved_dtype).copy(),
        ),
        previous_dimensionless_time=np.zeros(n, dtype=resolved_dtype),
        previous_aquifer_pressure=typing.cast(
            NumberArray[OneDimension],
            np.asarray(compiled.initial_pressures, dtype=resolved_dtype).copy(),
        ),
    )
