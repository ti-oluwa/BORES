"""Boundary system for unstructured polyhedral reservoir grids."""

from bores.reservoir.boundary.aquifers import CarterTracyAquifer, FetkovichAquifer
from bores.reservoir.boundary.base import BoundaryCondition, BoundaryConditionType
from bores.reservoir.boundary.conditions import BoundaryConditions, BoundaryRegion
from bores.reservoir.boundary.factories import (
    make_axis_aligned_boundary_conditions,
    make_boundary_region,
)
from bores.reservoir.boundary.types import (
    ConstantFluxBoundary,
    ConstantPressureBoundary,
    ProductivityIndexBoundary,
)

__all__ = [
    "BoundaryCondition",
    "BoundaryConditionType",
    "BoundaryConditions",
    "BoundaryRegion",
    "CarterTracyAquifer",
    "ConstantFluxBoundary",
    "ConstantPressureBoundary",
    "FetkovichAquifer",
    "ProductivityIndexBoundary",
    "make_axis_aligned_boundary_conditions",
    "make_boundary_region",
]
