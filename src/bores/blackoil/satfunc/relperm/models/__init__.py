"""Relative permeability analytical models/tables multi-phase flow simulations."""

from .brooks_corey import (  # noqa: F401
    BrooksCoreyRelPermTable,
    compute_brooks_corey_complement_curve,
    compute_brooks_corey_oil_wet_system,
    compute_brooks_corey_phase_curve,
    compute_brooks_corey_relative_permeabilities,
    compute_brooks_corey_relative_permeability_derivatives,
    compute_brooks_corey_water_wet_system,
)
from .let import (  # noqa: F401
    LETParameters,
    LETThreePhaseRelPermTable,
    _let_relperm,
    compute_let_curve_slope_wrt_normalized_saturation,
    compute_let_declining_curve,
    compute_let_oil_wet_system,
    compute_let_phase_curve,
    compute_let_relative_permeabilities,
    compute_let_relative_permeability_derivatives,
    compute_let_water_wet_system,
)

__all__ = [
    "BrooksCoreyRelPermTable",
    "LETParameters",
    "LETThreePhaseRelPermTable",
    "compute_brooks_corey_relative_permeabilities",
    "compute_brooks_corey_relative_permeability_derivatives",
    "compute_let_relative_permeabilities",
    "compute_let_relative_permeability_derivatives",
]
