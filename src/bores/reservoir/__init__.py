"""
Reservoir characterization. `Reservoir` itself, its rock/region/temperature
inputs, boundary conditions, transmissibility, and state.
"""

from bores.reservoir.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundaryConditionType,
    BoundaryRegion,
    CarterTracyAquifer,
    ConstantFluxBoundary,
    ConstantPressureBoundary,
    FetkovichAquifer,
    ProductivityIndexBoundary,
    make_axis_aligned_boundary_conditions,
    make_boundary_region,
)
from bores.reservoir.faults import Fault
from bores.reservoir.model import Reservoir
from bores.reservoir.regions import Regions
from bores.reservoir.rock import (
    Permeability,
    Rock,
    RockCompressibility,
    RockCompressibilityTable,
    RockCompressibilityTables,
)
from bores.reservoir.state import (
    DepthTable,
    Equilibrium,
    EquilibriumRegion,
    Hysteresis,
    ReservoirState,
    load_equilibrium_regions,
)
from bores.reservoir.temperature import Temperature, TemperatureGradient, TemperatureTable
from bores.reservoir.transmissibility import (
    ConnectionTransmissibilities,
    compute_connection_transmissibilities,
)

__all__ = [
    "BoundaryCondition",
    "BoundaryConditionType",
    "BoundaryConditions",
    "BoundaryRegion",
    "CarterTracyAquifer",
    "ConnectionTransmissibilities",
    "ConstantFluxBoundary",
    "ConstantPressureBoundary",
    "DepthTable",
    "Equilibrium",
    "EquilibriumRegion",
    "Fault",
    "FetkovichAquifer",
    "Hysteresis",
    "Permeability",
    "ProductivityIndexBoundary",
    "Regions",
    "Reservoir",
    "ReservoirState",
    "Rock",
    "RockCompressibility",
    "RockCompressibilityTable",
    "RockCompressibilityTables",
    "Temperature",
    "TemperatureGradient",
    "TemperatureTable",
    "compute_connection_transmissibilities",
    "load_equilibrium_regions",
    "make_axis_aligned_boundary_conditions",
    "make_boundary_region",
]
