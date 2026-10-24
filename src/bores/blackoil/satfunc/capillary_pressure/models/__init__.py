"""Capillary pressure analytical models/tables for multi-phase flow simulations."""

from bores.types import Wettability  # noqa: F401

from .brooks_corey import (  # noqa: F401
    BrooksCoreyCapillaryPressureTable,
    compute_brooks_corey_capillary_pressure_derivatives,
    compute_brooks_corey_capillary_pressures,
    compute_brooks_corey_capillary_pressures_array,
    compute_brooks_corey_capillary_pressures_scalar,
    compute_brooks_corey_derivatives_array,
    compute_brooks_corey_derivatives_scalar,
)
from .leverett_j import (  # noqa: F401
    LeverettJCapillaryPressureTable,
    compute_leverett_j_capillary_pressures,
    compute_leverett_j_capillary_pressures_array,
    compute_leverett_j_capillary_pressures_scalar,
    compute_leverett_j_derivatives,
    compute_leverett_j_derivatives_array,
    compute_leverett_j_derivatives_scalar,
    convert_dyne_per_cm_to_pressure_unit,
)
from .van_genuchten import (  # noqa: F401
    VanGenuchtenCapillaryPressureTable,
    compute_van_genuchten_capillary_pressures,
    compute_van_genuchten_capillary_pressures_array,
    compute_van_genuchten_capillary_pressures_scalar,
    compute_van_genuchten_derivatives,
    compute_van_genuchten_derivatives_array,
    compute_van_genuchten_derivatives_scalar,
    compute_van_genuchten_pc_slope_wrt_effective_saturation,
)

__all__ = [
    "BrooksCoreyCapillaryPressureTable",
    "LeverettJCapillaryPressureTable",
    "VanGenuchtenCapillaryPressureTable",
]
