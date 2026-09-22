"""Analytic aquifer boundary conditions."""

from bores.reservoir.boundary.aquifers.carter_tracy import CarterTracyAquifer
from bores.reservoir.boundary.aquifers.fetkovich import FetkovichAquifer

__all__ = ["CarterTracyAquifer", "FetkovichAquifer"]
