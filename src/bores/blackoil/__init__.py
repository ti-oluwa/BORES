"""Black-oil simulation model: `BlackOilModel`, its compiled form, and its state."""

from bores.blackoil.compile import CompiledBlackOilModel, compile_model
from bores.blackoil.fluids import BlackOil, Fluid
from bores.blackoil.model import BlackOilModel
from bores.blackoil.pseudo_pressure import PseudoPressureTable, build_pseudo_pressure_table
from bores.blackoil.state import BlackOilModelState

__all__ = [
    "BlackOil",
    "BlackOilModel",
    "BlackOilModelState",
    "CompiledBlackOilModel",
    "Fluid",
    "PseudoPressureTable",
    "build_pseudo_pressure_table",
    "compile_model",
]
