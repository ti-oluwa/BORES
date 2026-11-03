"""Relative permeability analytical models/tables multi-phase flow simulations."""

from .brooks_corey import (
    BrooksCoreyRelPermTable,
    compute_brooks_corey_relative_permeabilities,
    compute_brooks_corey_relative_permeability_derivatives,
)
from .let import (
    LETParameters,
    LETThreePhaseRelPermTable,
    compute_let_relative_permeabilities,
    compute_let_relative_permeability_derivatives,
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
