"""
Grid factory classes for constructing `bores.grids.base.Grid` objects
from various source representations.

**Face winding convention**:

All factories produce faces whose vertices are ordered **counter-clockwise when
viewed from the owner cell** (``face_cell_indices[:, 0]``).  Under this
convention, the Newell normal produced by `bores.grid.grid._compute_face_geometry`
points **from owner toward neighbour** (i.e. outward for the owner cell).
Boundary faces carry ``neighbour_index == -1``.

**Coordinate system**:

The z-axis is positive **downward** (reservoir depth convention), matching
`bores.grids.base.Grid`.
"""

from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import make_corner_point_grid, rederive_corner_point_arrays
from bores.grids.factories.polyhedral import make_polyhedral_grid
from bores.grids.factories.voronoi import make_voronoi_grid

__all__ = [
    "make_cartesian_grid",
    "make_corner_point_grid",
    "make_polyhedral_grid",
    "make_voronoi_grid",
    "rederive_corner_point_arrays",
]
