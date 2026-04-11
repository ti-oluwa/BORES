"""Precomputed geometric face transmissibilities for structured 3D grids."""

import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.correlations.core import compute_harmonic_mean
from bores.models import RockPermeability
from bores.types import ThreeDimensionalGrid, ThreeDimensions

__all__ = ["FaceTransmissibilities", "build_face_transmissibilities"]


class FaceTransmissibilities(typing.NamedTuple):
    """
    Precomputed geometric face transmissibilities for all interior faces in x, y, z.

    Each array is shaped (nx, ny, nz). Entry [i, j, k] stores the transmissibility
    of the *forward* face:

        x: face between cell (i,j,k) and (i+1,j,k)   units: mD·ft (= mD·ft²/ft)
        y: face between cell (i,j,k) and (i,j+1,k)
        z: face between cell (i,j,k) and (i,j,k+1)

    The last slice in each direction has no forward neighbour and holds zeros.
    """

    x: ThreeDimensionalGrid
    y: ThreeDimensionalGrid
    z: ThreeDimensionalGrid


def build_face_transmissibilities(
    absolute_permeability: RockPermeability[ThreeDimensions],
    thickness_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    dtype: npt.DTypeLike = np.float64,
) -> FaceTransmissibilities:
    """
    Precompute geometric face transmissibilities T_geo = k_harmonic * A / L for
    every forward-facing interface in x, y, and z on the grid.

    Result arrays are shaped (nx, ny, nz) where (nx, ny, nz) = grid shape.
    Entry [i, j, k] is the transmissibility of the face between (i,j,k) and:
        T_x: (i+1, j,   k  )
        T_y: (i,   j+1, k  )
        T_z: (i,   j,   k+1)

    The last row/col/layer in each direction is set to zero (no forward neighbour).

    :param absolute_permeability: Absolute permeability object with x, y, z arrays (mD).
    :param thickness_grid: Cell thickness grid (ft).
    :param net_to_gross_grid: Net-to-gross ratio grid (fraction).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param dtype: NumPy dtype for output arrays.
    :return: `FaceTransmissibilities` named tuple with x, y, z arrays (mD·ft).
    """
    Tx, Ty, Tz = _compute_face_transmissibilities(
        Kx=absolute_permeability.x,
        Ky=absolute_permeability.y,
        Kz=absolute_permeability.z,
        thickness_grid=thickness_grid,
        net_to_gross_grid=net_to_gross_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        dtype=dtype,
    )
    return FaceTransmissibilities(x=Tx, y=Ty, z=Tz)


@numba.njit(parallel=True, cache=True)
def _compute_face_transmissibilities(
    Kx: ThreeDimensionalGrid,
    Ky: ThreeDimensionalGrid,
    Kz: ThreeDimensionalGrid,
    thickness_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    nx, ny, nz = Kx.shape

    Tx = np.zeros((nx, ny, nz), dtype=dtype)
    Ty = np.zeros((nx, ny, nz), dtype=dtype)
    Tz = np.zeros((nx, ny, nz), dtype=dtype)

    for i in numba.prange(nx - 1):  # type: ignore[attr-defined]
        for j in range(ny - 1):
            for k in range(nz - 1):
                cell_thickness = thickness_grid[i, j, k]

                # X face: (i,j,k) → (i+1,j,k)
                # k_harmonic uses the x-permeability (flow in x-direction)
                kx_harmonic = compute_harmonic_mean(
                    Kx[i, j, k] * net_to_gross_grid[i, j, k],
                    Kx[i + 1, j, k] * net_to_gross_grid[i + 1, j, k],
                )
                # Face area in x: Δy × harmonic(cell_thickness, h_i+1)
                # Flow length: Δx (cell-centre to cell-centre = Δx for uniform grid)
                east_thickness = thickness_grid[i + 1, j, k]
                east_harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, east_thickness
                )
                flow_area_x = cell_size_y * east_harmonic_thickness
                Tx[i, j, k] = kx_harmonic * flow_area_x / cell_size_x

                # Y face: (i,j,k) → (i,j+1,k)
                ky_harmonic = compute_harmonic_mean(
                    Ky[i, j, k] * net_to_gross_grid[i, j, k],
                    Ky[i, j + 1, k] * net_to_gross_grid[i, j + 1, k],
                )
                south_thickness = thickness_grid[i, j + 1, k]
                south_harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, south_thickness
                )
                flow_area_y = cell_size_x * south_harmonic_thickness
                Ty[i, j, k] = ky_harmonic * flow_area_y / cell_size_y

                # Z face: (i,j,k) → (i,j,k+1)
                kz_harmonic = compute_harmonic_mean(
                    Kz[i, j, k] * net_to_gross_grid[i, j, k],
                    Kz[i, j, k + 1] * net_to_gross_grid[i, j, k + 1],
                )
                bottom_thickness = thickness_grid[i, j, k + 1]
                bottom_harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, bottom_thickness
                )
                # For vertical: area = Δx × Δy, flow length = harmonic mean thickness
                flow_area_z = cell_size_x * cell_size_y
                Tz[i, j, k] = kz_harmonic * flow_area_z / bottom_harmonic_thickness

    return Tx, Ty, Tz
