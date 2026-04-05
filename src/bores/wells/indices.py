import itertools
import typing

import attrs

from bores.boundary_conditions import BoundaryConditions
from bores.datastructures import ContextFlag
from bores.models import RockPermeability
from bores.solvers.base import to_1D_index_interior_only
from bores.types import NDimensionalGrid, ThreeDimensions
from bores.wells.base import Well, Wells


@attrs.frozen
class PerforationIndex:
    """Well index for a single perforated cell."""

    cell: typing.Tuple[int, int, int]  # padded (i, j, k)
    """Perforated cell padded indices"""
    cell_1d_index: int
    """Perforated cell padded 1D index"""
    well_index: float
    """Perforated cell well index"""


@attrs.frozen
class WellIndices:
    """Computed well indices for all perforations of a single well."""

    well_name: str
    """Well name"""
    perforations: typing.Tuple[PerforationIndex, ...]
    """Tuple of `PerforationIndex` objects"""
    total_well_index: float
    """Sum total of all perforated cells' well indices"""

    def allocation_fraction(self, perforation: PerforationIndex) -> float:
        if self.total_well_index <= 0:
            return 1.0
        return perforation.well_index / self.total_well_index

    def __iter__(self) -> typing.Iterator[PerforationIndex]:
        return iter(self.perforations)


@attrs.frozen
class WellIndicesCache:
    """(Pre)computed well indices for all wells."""

    injection: typing.Dict[str, WellIndices]
    """Mapping of injection well names to their corresponding `WellIndices`"""
    production: typing.Dict[str, WellIndices]
    """Mapping of injection well names to their corresponding `WellIndices`"""


def build_well_indices_cache(
    wells: Wells[ThreeDimensions],
    absolute_permeability: RockPermeability,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_size_x: float,
    cell_size_y: float,
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pad_width: int = 1,
) -> WellIndicesCache:
    """
    Compute well indices for all perforated cells across all injection and production wells.

    Well indices depend only on static geometric and rock properties (permeability, cell
    dimensions, skin factor, wellbore radius, and boundary conditions), so they are
    invariant across time steps and can be computed once at simulation startup.

    The resulting cache eliminates redundant well index computations from the inner
    time-step loop in the pressure and saturation solvers.

    :param wells: `Wells` container holding all injection and production wells.
    :param thickness_grid: Padded 3D grid of cell thicknesses (ft). Must include ghost cells.
    :param absolute_permeability: Padded absolute permeability object with x, y, z component grids (mD).
    :param boundary_conditions: Model boundary conditions. Used to apply no-flow boundary
        corrections to the Peaceman effective drainage radius for wells at grid boundaries.
    :param cell_size_x: Cell size in the x-direction (ft).
    :param cell_size_y: Cell size in the y-direction (ft).
    :param cell_count_x: Total number of cells in the x-direction, including ghost cells.
    :param cell_count_y: Total number of cells in the y-direction, including ghost cells.
    :param cell_count_z: Total number of cells in the z-direction, including ghost cells.
    :param pad_width: Number of ghost cells used for grid padding. Well coordinates are
        offset by this amount when indexing into padded grids. Defaults to 1.
    :return: `WellIndicesCache` containing precomputed `WellIndices` for every injection
        and production well, keyed by well name.
    """
    grid_dims = (
        cell_count_x - 2 * pad_width,
        cell_count_y - 2 * pad_width,
        cell_count_z - 2 * pad_width,
    )
    pressure_bc = boundary_conditions["pressure"]

    def _well_indices(well: Well) -> WellIndices:
        perforations = []
        total_well_index = 0.0
        for start, end in well.perforating_intervals:
            for i, j, k in itertools.product(
                range(start[0] + pad_width, end[0] + pad_width + 1),
                range(start[1] + pad_width, end[1] + pad_width + 1),
                range(start[2] + pad_width, end[2] + pad_width + 1),
            ):
                well_index = well.get_well_index(
                    interval_thickness=(
                        cell_size_x,
                        cell_size_y,
                        thickness_grid[i, j, k],  # type: ignore
                    ),
                    permeability=(
                        absolute_permeability.x[i, j, k],
                        absolute_permeability.y[i, j, k],
                        absolute_permeability.z[i, j, k],
                    ),
                    skin_factor=well.skin_factor,
                    well_location=(i - pad_width, j - pad_width, k - pad_width),
                    grid_dimensions=grid_dims,
                    boundary_condition=pressure_bc,
                )
                cell_1d_index = to_1D_index_interior_only(
                    i, j, k, cell_count_x, cell_count_y, cell_count_z
                )
                perforations.append(
                    PerforationIndex(
                        cell=(i, j, k),
                        cell_1d_index=cell_1d_index,
                        well_index=well_index,
                    )
                )
                total_well_index += well_index
        return WellIndices(
            well_name=well.name,
            perforations=tuple(perforations),
            total_well_index=total_well_index,
        )

    return WellIndicesCache(
        injection={wells.name: _well_indices(wells) for wells in wells.injection_wells},
        production={
            wells.name: _well_indices(wells) for wells in wells.production_wells
        },
    )


update_well_indices = ContextFlag(False)
"""Flag to indicate if a well indices update is needed during a simulation run."""
