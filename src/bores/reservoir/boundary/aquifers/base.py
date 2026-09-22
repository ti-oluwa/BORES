"""Primitives shared by every analytic aquifer kind."""

import typing

import numba

from bores.types import CellArray, IntArray, Number, OneDimension

__all__ = ["compute_average_boundary_pressure"]


@numba.njit(cache=True)
def compute_average_boundary_pressure(
    owner_cells: IntArray[OneDimension], pressure: CellArray
) -> Number:
    """
    Average reservoir pressure over one aquifer's own boundary faces'
    owner cells - a plain mean, matching `CarterTracyAquifer`'s own
    original, validated convention exactly (not area-weighted).

    Reads only `pressure` (a plain cell-pressure array, from
    `ReservoirWorkspace`, never a rich `ReservoirState`) and one static,
    precomputed geometry array - no grid or reservoir object lookup, and
    no allocation.

    :param owner_cells: Shape `(n_faces,)`. This aquifer's own boundary
        faces' owner cells, precomputed once at compile time.
    :param pressure: Current cell pressures, indexed by `owner_cells`.
    :returns: Scalar mean pressure across `owner_cells`.
    """
    total = 0.0
    for i in range(owner_cells.shape[0]):
        total += pressure[owner_cells[i]]
    return typing.cast(Number, total / owner_cells.shape[0])
