"""
Wellbore hydraulics: shared primitives plus per-correlation models.

`compute_perforation_pressures`, `compute_segment_drop`, and
`compute_tubing_head_pressure` are defined once per correlation module
(`beggs_and_brill`, `hagedorn_brown`, `homogeneous`) and are not
re-exported here, since the three definitions collide on name. Import
them from their own submodule.
"""

from bores.wells.hydraulics.base import (
    PressureDrop,
    SurfaceFluidProperties,
    WellBoreModel,
    WellBoreModelOptions,
    compute_friction_factor,
    compute_hydrostatic_pressure,
    compute_mixture_density,
    compute_mixture_velocity,
    compute_mixture_viscosity,
    compute_segment_pressure_drop,
    compute_static_hydrostatic_drop,
    compute_static_mixture_density,
    compute_surface_mixture_density,
    compute_surface_mixture_viscosity,
    get_unit_system_constant,
)
from bores.wells.hydraulics.beggs_and_brill import (
    BeggsAndBrillModel,
    beggs_and_brill_model,
    compute_beggs_brill_holdup,
    compute_two_phase_friction_factor,
    flow_pattern_tag,
    horizontal_holdup,
)
from bores.wells.hydraulics.hagedorn_brown import (
    HagedornBrownModel,
    compute_griffith_holdup,
    compute_hagedorn_brown_holdup,
    hagedorn_brown_model,
    is_griffith_bubble_flow,
)
from bores.wells.hydraulics.homogeneous import HomogeneousModel, homogeneous_model

__all__ = [
    "BeggsAndBrillModel",
    "HagedornBrownModel",
    "HomogeneousModel",
    "PressureDrop",
    "SurfaceFluidProperties",
    "WellBoreModel",
    "WellBoreModelOptions",
    "beggs_and_brill_model",
    "compute_beggs_brill_holdup",
    "compute_friction_factor",
    "compute_griffith_holdup",
    "compute_hagedorn_brown_holdup",
    "compute_hydrostatic_pressure",
    "compute_mixture_density",
    "compute_mixture_velocity",
    "compute_mixture_viscosity",
    "compute_segment_pressure_drop",
    "compute_static_hydrostatic_drop",
    "compute_static_mixture_density",
    "compute_surface_mixture_density",
    "compute_surface_mixture_viscosity",
    "compute_two_phase_friction_factor",
    "flow_pattern_tag",
    "get_unit_system_constant",
    "hagedorn_brown_model",
    "homogeneous_model",
    "horizontal_holdup",
    "is_griffith_bubble_flow",
]
