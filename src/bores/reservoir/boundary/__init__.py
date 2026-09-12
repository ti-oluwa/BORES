"""Boundary system for unstructured polyhedral reservoir grids."""

from bores.reservoir.boundary.aquifers import CarterTracyAquifer
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
    TimeDependentFluxBoundary,
)

__all__ = [
    "BoundaryCondition",
    "BoundaryConditionType",
    "BoundaryConditions",
    "BoundaryRegion",
    "CarterTracyAquifer",
    "ConstantFluxBoundary",
    "ConstantPressureBoundary",
    "ProductivityIndexBoundary",
    "TimeDependentFluxBoundary",
    "make_axis_aligned_boundary_conditions",
    "make_boundary_region",
]
