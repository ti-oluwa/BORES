"""Relative permeability analytical models/tables multi-phase flow simulations."""

from .brooks_corey import (
    BrooksCoreyRelPermTable,
    compute_brookes_corey_relative_permeabilities,
)
from .let import (  # noqa: F401
    LETParameters,
    LETThreePhaseRelPermTable,
    _let_relperm,
    compute_let_curve_slope_wrt_normalized_saturation,
    compute_let_relative_permeabilities,
)

__all__ = [
    "BrooksCoreyRelPermTable",
    "LETParameters",
    "LETThreePhaseRelPermTable",
    "compute_brookes_corey_relative_permeabilities",
    "compute_let_relative_permeabilities",
]
